#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import logging
import os
import json
import numpy as np
import random
from collections import Counter
import time
from contextlib import nullcontext
from pprint import pformat
import torch.distributed as dist
from typing import Any, Mapping, Sequence, List, Dict

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import DistributedEpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)

import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from typing import Any

def extract_episode_indices(batch):
    ep = batch.get("episode_index", None)
    if ep is None:
        return []

    if torch.is_tensor(ep):
        return [int(x) for x in ep.detach().cpu().view(-1).tolist()]

    if isinstance(ep, (list, tuple)):
        out = []
        for x in ep:
            out.append(int(x.item()) if torch.is_tensor(x) else int(x))
        return out

    return [int(ep)]

def jain_fairness_index(counter, num_episodes: int) -> float:
    counts = np.array(
        [counter.get(i, 0) for i in range(num_episodes)],
        dtype=np.float64,
    )

    total = counts.sum()
    if total == 0:
        return float("nan")

    denom = num_episodes * np.sum(counts ** 2)
    if denom == 0:
        return float("nan")

    return float((total ** 2) / denom)

def simple_collate(batch: Sequence[Any]) -> Any:
    """
    default_collate の代わりに使うシンプルな collate_fn。
    - Tensor: torch.stack でまとめる（共有メモリを強制しない）
    - dict:  再帰的にキーごとに collate
    - list/tuple: 転置して再帰的に collate
    - int/float/bool: torch.tensor に変換
    - それ以外: そのままリストとして返す
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]

    # Tensor
    if torch.is_tensor(elem):
        return torch.stack(batch, dim=0)

    # dict
    if isinstance(elem, Mapping):
        return {k: simple_collate([d[k] for d in batch]) for k in elem}

    # list / tuple
    if isinstance(elem, (list, tuple)):
        transposed = list(zip(*batch))
        return [simple_collate(samples) for samples in transposed]

    # スカラ
    if isinstance(elem, (int, float, bool)):
        return torch.tensor(batch)

    # 文字列などはそのまま
    return batch

class SequenceLeRobotDataset(Dataset):
    """
    LeRobotDataset をラップして CoTVLA 仕様の時系列サンプルを作るラッパー。

    - キー名・構造は base_dataset と同じ
      例: 'observation.images.front', 'action', 'observation.state', ...

    - 観測系: idx から n_obs_steps フレーム分を [T_obs, ...] でスタック
    - action: 観測直後から n_action_steps フレーム分を [T_action, ...] でスタック
      → 未来アクションのみを教師として使う CoTVLA 仕様

    - subgoal 観測:
      - 未来アクションフレームのうち先頭 n_subgoal_steps 分の
        'observation.images.front' / 'observation.images.wrist' を
        それぞれ
          'future_observation.images.front'
          'future_observation.images.wrist'
        として [T_subgoal, C, H, W] で返す
      - CoTVLA の image reconstruction (Visual CoT) 用の target になる想定

    - さらに:
      - 動画デコード時に AssertionError が出るサンプルはスキップ
      - front / wrist どちらかの画像が欠けている or NaN を含むシーケンスもスキップ
    """

    def __init__(
        self,
        base_dataset: Dataset,
        n_obs_steps: int,
        n_action_steps: int = 2,      # CoTVLA 論文では K=2 が基本
        n_subgoal_steps: int = 1,     # subgoal 観測フレーム数（デフォルト 1）
        max_retry: int = 10,          # 壊れたサンプルに当たったときのリトライ回数
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_subgoal_steps = n_subgoal_steps
        self.max_retry = max_retry

        # subgoal は future_action_frames の先頭から取る前提にする
        assert (
            self.n_subgoal_steps <= self.n_action_steps
        ), "n_subgoal_steps は n_action_steps 以下にしてください"

        # 1ステップで実際に参照するフレーム数（観測 + 未来アクション）
        self.window_size = self.n_obs_steps + self.n_action_steps

        # 🔴 メタ情報ではなく「実際の base_dataset の長さ」を唯一のソースにする
        self.num_transitions = len(self.base_dataset)

        # 取りうる開始 idx の総数
        # 例: base_len=100, window_size=12 → 0〜88 までの 89 通り
        self.length = max(0, self.num_transitions - self.window_size + 1)
        
        # マスクキー一覧（base_dataset の meta から取得）
        self.mask_weights = getattr(base_dataset, "mask_weights", {})

        # 期待するマスクのフラットキー（LeRobotDataset.__getitem__ がこれを返す想定）
        self.required_mask_keys = []
        for cam in ["observation.images.front", "observation.images.wrist"]:
            for mk in self.mask_weights.keys():
                if mk == "background":
                    continue
                self.required_mask_keys.append(f"{cam}/mask/{mk}")


    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _stack_over_time(sample0: Any, frames_values: List[Any]) -> Any:
        """
        観測側（obs）のための再帰スタッカー:
        - torch.Tensor → [T_obs, ...] に stack
        - dict        → 再帰的にキーごとに _stack_over_time
        - その他      → 最後のフレームの値をそのまま返す
        """
        if torch.is_tensor(sample0):
            return torch.stack(frames_values, dim=0)

        if isinstance(sample0, dict):
            out: Dict[str, Any] = {}
            for k, v0 in sample0.items():
                v_list = [fv[k] for fv in frames_values]
                out[k] = SequenceLeRobotDataset._stack_over_time(v0, v_list)
            return out

        # str, float, int, None などは最後の値だけ使う
        return frames_values[-1]

    def _is_invalid_frame(self, frame: Dict[str, Any]) -> bool:
        required_img_keys = [
            "observation.images.front",
            "observation.images.wrist",
        ]

        # 画像チェック
        for key in required_img_keys:
            if key not in frame:
                return True
            img = frame[key]
            if isinstance(img, torch.Tensor) and torch.isnan(img).any():
                return True

        # マスクチェック（mask_keys がある場合のみ）
        if hasattr(self, "required_mask_keys"):
            for mkey in self.required_mask_keys:
                if mkey not in frame:
                    return True
                m = frame[mkey]
                if not isinstance(m, torch.Tensor):
                    return True
                if torch.isnan(m.float()).any():
                    return True

        return False
    
    def _get_ep(self, frame):
        ep = frame["episode_index"]
        return int(ep.item()) if torch.is_tensor(ep) else int(ep)

    def _build_seq_sample(
        self,
        idx: int,
        obs_frames: List[Dict[str, Any]],
        future_action_frames: List[Dict[str, Any]],
        future_obs_frames: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """元の __getitem__ 本体部分を切り出したヘルパ。"""
        sample0 = obs_frames[0]
        seq_sample: Dict[str, Any] = {}
        
        ep0 = self._get_ep(obs_frames[0])
        for f in obs_frames + future_action_frames:
            if self._get_ep(f) != ep0:
                raise AssertionError(f"Cross-episode sequence detected at idx={idx}: ep0={ep0}")
        seq_sample["episode_index"] = ep0
        
        for key, v0 in sample0.items():
            # action だけは「未来フレーム」から T_action を作る
            if key == "action":
                action_vals = [f[key] for f in future_action_frames]
                if torch.is_tensor(v0):
                    # [1, ...] → [...] にしてから stack（DataCollator 対応）
                    action_vals = [a.squeeze(0) for a in action_vals]
                    seq_sample[key] = torch.stack(action_vals, dim=0)  # [T_action, ...]
                else:
                    seq_sample[key] = action_vals[-1]
                continue

            # 観測系その他は obs_frames から [T_obs, ...] にスタック
            vals = [f[key] for f in obs_frames]
            seq_sample[key] = self._stack_over_time(v0, vals)

        # --- subgoal 観測画像 (future_observation.*) を追加 ---
        #   - future_obs_frames のうち、カメラ画像だけを取り出して stack
        for cam_key in ["observation.images.front", "observation.images.wrist"]:
            if cam_key in future_obs_frames[0]:
                imgs = [f[cam_key] for f in future_obs_frames]
                if torch.is_tensor(imgs[0]):
                    # [T_subgoal, C, H, W]
                    seq_sample[f"future_{cam_key}"] = torch.stack(imgs, dim=0)
                    
        # --- subgoal 観測マスク (future_.../mask/...) を追加 ---
        if hasattr(self, "mask_weights") and len(self.mask_weights) > 0:
            for cam_key in ["observation.images.front", "observation.images.wrist"]:
                for mk in self.mask_weights.keys():
                    if mk == "background":
                        continue
                    k = f"{cam_key}/mask/{mk}"
                    if k in future_obs_frames[0]:
                        ms = [f[k] for f in future_obs_frames]
                        if torch.is_tensor(ms[0]):
                            # [T_subgoal, H, W]
                            seq_sample[f"future_{k}"] = torch.stack(ms, dim=0)

        # メタ情報
        seq_sample["index_start"] = idx
        seq_sample["obs_len"] = self.n_obs_steps
        seq_sample["action_len"] = self.n_action_steps
        seq_sample["subgoal_len"] = self.n_subgoal_steps

        return seq_sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        idx: 観測シーケンスの開始フレーム index

        戻り値:
        - base_dataset と同じキー構造
        - ただしテンソルは:
            観測系: [T_obs, ...]
            action: [T_action, ...] （未来アクションのみ）
            future_observation.images.front : [T_subgoal, C, H, W]
            future_observation.images.wrist : [T_subgoal, C, H, W]
        """

        original_idx = idx  # デバッグ用に保持

        for attempt in range(self.max_retry):
            # 念のためガード（DataLoader のバグ混入や length 計算ミス検出用）
            if idx < 0 or idx + self.window_size > self.num_transitions:
                # おかしな idx の場合はランダムに振り直して再試行
                idx = random.randrange(self.length)
                continue

            try:
                # 観測フレーム: idx 〜 idx+n_obs_steps-1
                obs_frames = [self.base_dataset[idx + t] for t in range(self.n_obs_steps)]

                # 未来アクション用フレーム: 観測の直後から n_action_steps 個
                future_action_frames = [
                    self.base_dataset[idx + self.n_obs_steps + t]
                    for t in range(self.n_action_steps)
                ]

                # subgoal 観測用フレーム（未来アクションの末尾 n_subgoal_steps を再利用）
                future_obs_frames = future_action_frames[-self.n_subgoal_steps:]

            except AssertionError as e:
                # decode_video_frames_torchvision の AssertionError など
                idx = random.randrange(self.length)
                continue

            # どれか 1 フレームでも「どちらかのカメラが NaN / 無い」ならこのシーケンスは捨てる
            bad_obs = any(self._is_invalid_frame(f) for f in obs_frames)
            bad_future = any(self._is_invalid_frame(f) for f in future_action_frames)
            if bad_obs or bad_future:
                idx = random.randrange(self.length)
                continue
            
            ep0 = self._get_ep(obs_frames[0])
            if any(self._get_ep(f) != ep0 for f in (obs_frames + future_action_frames)):
                idx = random.randrange(self.length)
                continue

            # ここまで来たら「まともなシーケンス」とみなして組み立てて返す
            try:
                return self._build_seq_sample(idx, obs_frames, future_action_frames, future_obs_frames)
            except AssertionError:
                # Cross-episode などは捨ててリトライ
                idx = random.randrange(self.length)
                continue

        # 何度リトライしてもダメだった場合はエラーにする
        raise RuntimeError(
            f"Too many failed samples in SequenceLeRobotDataset.__getitem__ "
            f"(orig_idx={original_idx}, max_retry={self.max_retry})"
        )
        
def to_csv_cell(x):
    # torch.Tensor
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return float(x.item())
        # 多要素は list 化して JSON で1セルに入れる
        return json.dumps(x.tolist(), ensure_ascii=False)

    # numpy
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        return json.dumps(x.tolist(), ensure_ascii=False)

    # python scalar
    if isinstance(x, (int, float, bool, np.number)):
        return float(x)

    # list/tuple/dict は JSON 文字列で保存
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(x, ensure_ascii=False)

    # その他（例: None, str など）
    if x is None:
        return ""
    return str(x)

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    mask_weights: Dict[str, float]={},
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.total_loss = loss.item()
    train_metrics.action_loss = output_dict.get("action_loss", 0.0)
    train_metrics.recon_loss = output_dict.get("recon_loss", 0.0)
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    
    if mask_weights != {}:
        for mk in mask_weights.keys():
            v = output_dict.get(f"recon_loss/{mk}", None)
            if v is not None:
                train_metrics.update_metric(f"recon_loss/{mk}", float(v))
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    n_obs_steps = cfg.policy.n_obs_steps
    n_action_steps = getattr(cfg.policy, "n_action_steps", 1)
    logging.info(f"n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}")

    # シーケンス対応 Dataset にラップ
    seq_dataset = SequenceLeRobotDataset(
        base_dataset=dataset,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )

    is_ddp = dist.is_available() and dist.is_initialized()

    horizon = n_obs_steps + n_action_steps - 1
    drop_n_last_frames = max(cfg.policy.drop_n_last_frames, horizon)

    sampler = DistributedEpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        drop_n_first_frames=getattr(cfg.policy, "drop_n_first_frames", 0),
        drop_n_last_frames=drop_n_last_frames,
        horizon=horizon,
        shuffle=True,
        seed=42,
    )

    raw_dataloader = torch.utils.data.DataLoader(
        seq_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        shuffle=False,
        drop_last=is_ddp,          # ★DDPはTrue推奨
        pin_memory=True,
        timeout=0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=simple_collate,
    )
    
    steps_per_epoch = len(raw_dataloader) 
    
    epoch = step // steps_per_epoch
    step_in_epoch = step % steps_per_epoch

    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, raw_dataloader, lr_scheduler
    )
    dl_iter = iter(dataloader)
    
    if len(dataloader) == 0:
        raise RuntimeError("Empty dataloader: check drop_last/batch_size/sampler filtering.")

    for _ in range(step_in_epoch):
        try:
            next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)  # or raise
            next(dl_iter)
    
    policy.train()

    train_metrics = {
        "action_loss": AverageMeter("act_loss", ":.3f"),
        "recon_loss": AverageMeter("recon_loss", ":.3f"),
        "total_loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    
    if hasattr(cfg.policy, "mask_weights"):
        if cfg.policy.mask_weights != {}:
            for mk in cfg.policy.mask_weights.keys():
                key = f"recon_loss/{mk}"
                train_metrics[key] = AverageMeter(key, ":.3f")

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
        csv_path=os.path.join(cfg.output_dir, "train_output_log.csv"),
        csv_interval=200,
        is_main_process=is_main_process,
        extract_episode_indices_fn=extract_episode_indices,
        to_csv_cell_fn=to_csv_cell,
        jain_fairness_index_fn=jain_fairness_index,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
        logging.info("len(dataloader)=%d", len(dataloader)) 
        
    try:
        for global_step in tqdm(range(step, cfg.steps), desc="Training steps", total=cfg.steps - step):

            start_time = time.perf_counter()
            
            # このepoch内の最後で全rank同時に巻き戻す
            if global_step != step and (global_step % steps_per_epoch) == 0:
                epoch += 1
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                dl_iter = iter(dataloader)

            try:
                batch = next(dl_iter)
            except StopIteration:
                if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(step)

                dl_iter = iter(dataloader)

                try:
                    batch = next(dl_iter)
                except StopIteration:
                    # dataloaderが空（= 1バッチも作れない）
                    raise RuntimeError(
                        "Dataloader is empty after reinitialization. "
                        "Likely causes: DistributedSampler(drop_last=True) with too few samples, "
                        "or dataset filtering produced zero valid sequences on this rank."
                    )
                            
            train_tracker.on_batch(batch)
            batch = preprocessor(batch)

            train_tracker.dataloading_s = time.perf_counter() - start_time

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
                mask_weights=cfg.policy.mask_weights if hasattr(cfg.policy, "mask_weights") else {}
            )

            # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
            # increment `step` here.
            step += 1
            train_tracker.step()
            train_tracker.maybe_write_csv(step=step)

            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                if is_main_process:
                    logging.info(f"Checkpoint policy after step {step}")
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    update_last_checkpoint(checkpoint_dir)
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)

                accelerator.wait_for_everyone()

            if cfg.env and is_eval_step:
                if is_main_process:
                    step_id = get_step_identifier(step, cfg.steps)
                    with torch.no_grad(), accelerator.autocast():
                        eval_info = eval_policy_all(
                            envs=eval_env,  # dict[suite][task_id] -> vec_env
                            policy=accelerator.unwrap_model(policy),
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            n_episodes=cfg.eval.n_episodes,
                            videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                            max_episodes_rendered=4,
                            start_seed=cfg.seed,
                            max_parallel_tasks=cfg.env.max_parallel_tasks,
                        )
                    # overall metrics (suite-agnostic)
                    aggregated = eval_info["overall"]

                    # optional: per-suite logging
                    for suite, suite_info in eval_info.items():
                        logging.info("Suite %s aggregated: %s", suite, suite_info)

                    # meters/tracker
                    eval_metrics = {
                        "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                        "pc_success": AverageMeter("success", ":.1f"),
                        "eval_s": AverageMeter("eval_s", ":.3f"),
                    }
                    eval_tracker = MetricsTracker(
                        cfg.batch_size,
                        dataset.num_frames,
                        dataset.num_episodes,
                        eval_metrics,
                        initial_step=step,
                        accelerator=accelerator,
                    )
                    eval_tracker.eval_s = aggregated.pop("eval_s")
                    eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                    eval_tracker.pc_success = aggregated.pop("pc_success")
                    if wandb_logger:
                        wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                        wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

                accelerator.wait_for_everyone()
    except Exception as e:
        print("FATAL:", repr(e))
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
