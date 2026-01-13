#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
CoTVLA

Designed by Hugging Face.

Install cotvla extra dependencies:
```bash
pip install -e ".[cotvla]"
```

Example of finetuning the cotvla pretrained model (`cotvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/cotvla_base \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a cotvla. cotvla is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=cotvla \
--dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the cotvla pretrained model outside LeRobot training framework:
```python
policy = cotvlaPolicy.from_pretrained("lerobot/cotvla_base")
```

"""

import math
from collections import deque
import os
from typing import Any, Deque, Dict, List, Optional, Tuple

from termcolor import colored
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.cotvla.configuration_cotvla import CoTVLAConfig
from lerobot.policies.cotvla.cotvla_model import CoTVLA
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def resize_with_pad_mask(mask, width, height):
    """
    mask: [B, 1, H, W]  (float / uint8 / bool OK)
    return: [B, 1, height, width]
    """
    if mask.ndim != 4:
        raise ValueError(f"(b,1,h,w) expected, but {mask.shape}")

    cur_height, cur_width = mask.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # ★ nearest に変更
    resized_mask = F.interpolate(
        mask.float(),
        size=(resized_height, resized_width),
        mode="nearest",
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # ★ pad_value = 0
    padded_mask = F.pad(resized_mask, (pad_width, 0, pad_height, 0), value=0.0)

    return padded_mask


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class CoTVLAPolicy(PreTrainedPolicy):
    """Wrapper around CoTVLA model to train and run inference within LeRobot.

    CoTVLA の内部では
      - 画像 + 言語 + 状態から世界知識 (dynamic region / depth / semantics etc.) を予測
      - 世界知識を介して拡散 Transformer でアクション分布をモデリング
    するが、Policy の外側のインターフェイスは SmolVLA とほぼ同じにしておく。
    """

    config_class = CoTVLAConfig
    name = "cotvla"

    def __init__(self, config: CoTVLAConfig):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # 公式実装に基づいた CoTVLA 本体
        print("device:", config.device)
        self.model = CoTVLA(
            clip_device=config.device,
            vit_checkpoint_path=config.vit_checkpoint_path,
            n_obs_steps=config.n_obs_steps,
            num_resampler_query=config.num_resampler_query,
            num_obs_token_per_image=config.num_obs_token_per_image,
            atten_only_obs=config.atten_only_obs,
            attn_robot_proprio_state=config.attn_robot_proprio_state,
            atten_goal=config.atten_goal,
            atten_goal_state=config.atten_goal_state,
            mask_l_obs_ratio=config.mask_l_obs_ratio,
            resize_imgs_with_padding=config.resize_imgs_with_padding,
            patch_size=config.patch_size,
            n_action_steps=config.n_action_steps,
            transformer_layers=config.transformer_layers,
            hidden_dim=config.hidden_dim,
            transformer_heads=config.transformer_heads,
            phase=config.phase,
            pred_num=config.pred_num,
            use_gpt2_pretrained=config.use_gpt2_pretrained,
            attn_implementation=config.attn_implementation,
            obs_pred=config.obs_pred,
            use_sam_mask_loss=config.use_sam_mask_loss,
        )
        self.model.to(config.device)
        self.reset()

    # ------------------------------------------------------------------
    #  State & optimizer
    # ------------------------------------------------------------------
    def reset(self):
        """環境 reset ごとに呼ぶ。アクションのチャンク用キューをリセット。"""
        self._queues: Dict[str, Deque[Tensor]] = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        """Trainer から optimizer を作るときに使うパラメータ集合。"""
        return self.parameters()

    # ------------------------------------------------------------------
    #  Core chunked action selection
    # ------------------------------------------------------------------
    def _get_action_chunk(
        self,
        batch: Dict[str, Tensor],
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """(B, T, ·) のバッチから (B, n_action_steps, action_dim) のアクション列を生成。"""
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        state = state.unsqueeze(1)  # (B, 1, state_dim)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # CoTVLA 本体でアクションをサンプリング
        arm_pred_action, _, image_pred, _ = self.model.forward(
            image_primary=images[1],
            image_wrist=images[0],
            state=state,
            text_token=lang_tokens,
            text_attn=lang_masks,
            mode="inference",
            future_image_primary=images[2] if len(images) > 2 else None,
            future_image_wrist=images[3] if len(images) > 3 else None,
        )
        actions = arm_pred_action  # (B, n_action_steps, action_dim)

        # パディングした action_dim を元のサイズに戻す
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        # もし pi_aloha 互換モードが必要ならここで変換
        if getattr(self.config, "adapt_to_pi_aloha", False):
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """環境固有（pi_aloha など）の前処理をまとめる。"""
        if getattr(self.config, "adapt_to_pi_aloha", False):
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        return batch

    # ------------------------------------------------------------------
    #  Inference: multi-step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: Dict[str, Tensor],
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """(B, T, ·) から (B, n_action_steps, action_dim) を一度に予測したいとき用。"""
        self.eval()

        batch = self._prepare_batch(batch)
        # ACTION 以外の履歴をキューに詰めたい場合のためのヘルパ
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise)
        return actions

    # ------------------------------------------------------------------
    #  Inference: single-step（環境実行用）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(
        self,
        batch: Dict[str, Tensor],
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """環境からの観測に対して 1 ステップ分のアクションを返す。

        内部では n_action_steps 分のアクションを一括でサンプルしてキューに溜め、
        1 ステップずつ取り出す。
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            # (B, T, A) → (T, B, A) にして、T 分をキューに詰める
            self._queues[ACTION].extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]  # type: ignore
            )

        return self._queues[ACTION].popleft()

    # ------------------------------------------------------------------
    #  Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """トレーニング時の forward (損失計算)"""

        if getattr(self.config, "adapt_to_pi_aloha", False):
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)

        # -----------------------------
        # masks: mask_weights が dict で非空なら有効
        # -----------------------------
        mask_weights_cfg = getattr(self.config, "mask_weights", None)
        has_mask_weights = isinstance(mask_weights_cfg, dict) and (len(mask_weights_cfg) > 0)

        masks_dict, pad_dict, mask_keys = {}, {}, []
        mask_num_classes = None

        if has_mask_weights:
            masks_dict, pad_dict, mask_keys, meta = self.prepare_masks_multiclass_dict(batch)

            class_map = getattr(self, "mask_class_map", {}) or {}
            max_cid = 0
            for mk in mask_keys:
                max_cid = max(max_cid, int(class_map.get(mk, 0)))
            mask_num_classes = max(1, int(max_cid + 1))  # bg(0)込み

        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        loss_dict: Dict[str, Tensor] = {}

        # -----------------------------
        # CoTVLA 本体の forward
        # -----------------------------
        _, loss_arm_action, _, image_recon_error = self.model.forward(
            image_primary=images[1],
            image_wrist=images[0],
            state=state,
            text_token=lang_tokens,
            text_attn=lang_masks,
            action_label=actions,
            mode="train",
            future_image_primary=images[2] if len(images) > 2 else None,
            future_image_wrist=images[3] if len(images) > 3 else None,
            masks_dict=masks_dict if len(masks_dict) > 0 else None,
            mask_num_classes=mask_num_classes,
        )

        if loss_arm_action is None:
            loss_arm_action = torch.tensor(0.0, device=images[0].device)

        device = loss_arm_action.device

        # -----------------------------
        # recon loss
        # -----------------------------
        if image_recon_error is not None:
            class_map = getattr(self, "mask_class_map", {}) or {}
            class_weight_by_id = getattr(self, "mask_class_weight_by_id", None)

            recon_loss_weighted, recon_logs = self._aggregate_recon_loss_weighted(
                image_recon_error=image_recon_error,
                device=device,
                mask_keys=mask_keys,
                class_map=class_map,
                mask_weights_cfg=mask_weights_cfg,
                class_weight_by_id=class_weight_by_id,
                include_background=True,
                background_key="background",
                count_missing_as_zero=False,  # 旧挙動に寄せるなら True
            )

            loss_dict.update(recon_logs)

            losses = loss_arm_action + float(self.config.img_recon_loss_weight) * recon_loss_weighted

        else:
            loss_dict["recon_loss/all"] = torch.tensor(0.0, device=device)
            loss_dict["recon_loss"] = torch.tensor(0.0, device=device)
            losses = loss_arm_action

        # アクション予測誤差
        loss_dict["action_loss"] = loss_arm_action

        # 最終的な平均化（Tensorで返す方がログが安定）
        loss = losses.mean()
        loss_dict["loss"] = loss

        return loss, loss_dict

    # ------------------------------------------------------------------
    #  Feature preparation helpers
    # ------------------------------------------------------------------
    def prepare_images(self, batch: Dict[str, Tensor]):
        """CoTVLA 用の画像前処理。

        - (B, T, C, H, W) の場合は **全フレーム T を使用**(CoTVLA 仕様)
        - (B, C, H, W) の場合は T=1 として扱う
        - 必要ならリサイズ＋パディングでアスペクト保持
        - [0, 1] → [-1, 1] にスケーリング
        """
        images: List[Tensor] = []
        img_masks: List[Tensor] = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"At least one expected. (batch keys: {batch.keys()}) "
                f"(image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            x = batch[key]  # 期待: [B, T, C, H, W] or [B, C, H, W]
            # print(f"Preparing image feature '{key}' with shape {x.shape}")

            if x.ndim == 4:
                # [B, C, H, W] → [B, 1, C, H, W] として扱う
                x = x.unsqueeze(1)

            if x.ndim != 5:
                raise ValueError(f"{key} must be 4D or 5D tensor, got shape {x.shape}")

            B, T, C, H, W = x.shape

            # --- リサイズ＋パディング ---
            if self.config.resize_imgs_with_padding is not None:
                rh, rw = self.config.resize_imgs_with_padding

                # (B*T, C, H, W) にフラット化して一括処理
                x_flat = x.reshape(B * T, C, H, W)
                x_flat = resize_with_pad(x_flat, rh, rw, pad_value=0)  # [B*T, C, rh, rw]
                x = x_flat.reshape(B, T, C, rh, rw)

            # [0, 1] → [-1, 1]
            x = x * 2.0 - 1.0

            device = x.device

            # padding mask（カメラ有無のマスクなのでとりあえず [B] or [B, T] どちらでもOK）
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
                # もし 1D の [B] しかなければ、[B, T] にブロードキャストしてもよい
                if mask.ndim == 1:
                    mask = mask.unsqueeze(1).expand(B, T)  # [B, T]
            else:
                # 全フレーム有効とみなす
                mask = torch.ones(B, T, dtype=torch.bool, device=device)

            images.append(x)      # [B, T, C, H, W]
            img_masks.append(mask)  # [B, T]

        # 画像が欠けているカメラ用に「空のカメラ」を追加（SmolVLA と同じ処理）
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            # 既存カメラと同じ形で真っ黒画像を作る
            img = torch.ones_like(images[0]) * -1  # [-1, ...]
            mask = torch.zeros_like(img_masks[0])  # [B, T] 全部無効
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
    
    def build_mask_class_map(self, mask_keys: list[str]) -> dict[str, int]:
        # 0 は背景固定、1..K を mask_key に割り当て
        return {mk: i + 1 for i, mk in enumerate(mask_keys)}
    
    def extract_mask_key(self, feature_key: str) -> str:
        # 例: "observation.images.front/mask/sam" -> "sam"
        parts = feature_key.split("/mask/")
        if len(parts) != 2:
            raise ValueError(f"feature key does not contain '/mask/': {feature_key}")
        return parts[1]
    
    def _to_bt1hw(self, m: Tensor) -> Tuple[Tensor, int, int, int, int]:
        """
        任意形状の mask を [B,T,1,H,W] に正規化して返す（値はそのまま）
        対応:
        [B,H,W]
        [B,1,H,W]
        [B,T,H,W]
        [B,T,1,H,W]
        [B,T,N,H,W] -> ここでは OR 合成して [B,T,1,H,W] にする（必要なら別処理）
        """
        if m.ndim == 3:
            m = m.unsqueeze(1).unsqueeze(2)          # [B,1,1,H,W]
        elif m.ndim == 4:
            if m.shape[1] == 1:
                m = m.unsqueeze(1)                   # [B,1,1,H,W]
            else:
                m = m.unsqueeze(2)                   # [B,T,1,H,W]
        elif m.ndim == 5:
            pass
        else:
            raise ValueError(f"mask must be 3D/4D/5D, got {m.shape}")

        B, T, K, H, W = m.shape
        if K != 1:
            m = (m > 0).any(dim=2, keepdim=True)     # [B,T,1,H,W]
        return m, B, T, H, W
    
    
    def _get_weight_for_key(
        self,
        *,
        mk: str,
        cid: int,
        mask_weights_cfg: Optional[dict],
        class_weight_by_id: Optional[dict],
        default_w: float = 1.0,
    ) -> float:
        """
        重みの優先順位:
        1) class_weight_by_id[cid] (cid>0 のときのみ)
        2) mask_weights_cfg[mk]
        3) default_w
        """
        if cid > 0 and isinstance(class_weight_by_id, dict):
            return float(class_weight_by_id.get(cid, default_w))
        if isinstance(mask_weights_cfg, dict):
            return float(mask_weights_cfg.get(mk, default_w))
        return float(default_w)


    def _is_rank0(self) -> bool:
        # accelerate / torchrun のどっちでも効く簡易判定
        print("RANK:", os.environ.get("RANK", "N/A"), "LOCAL_RANK:", os.environ.get("LOCAL_RANK", "N/A"))
        return int(os.environ.get("RANK", "0")) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0

    def _to_float(self, x: Tensor) -> float:
        return float(x.detach().float().mean().item())

    def _aggregate_recon_loss_weighted(
        self,
        *,
        image_recon_error: dict,
        device: torch.device,
        mask_keys: List[str],
        class_map: dict,
        mask_weights_cfg: Optional[dict],
        class_weight_by_id: Optional[dict],
        include_background: bool = True,
        background_key: str = "background",
        count_missing_as_zero: bool = False,
        # --- debug ---
        debug: bool = False,
        step: Optional[int] = None,
        debug_every: int = 5,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:

        logs: Dict[str, Tensor] = {}
        all_mse = image_recon_error.get("all_batch_mse", None)
        all_mse = torch.tensor(0.0, device=device) if all_mse is None else all_mse.to(device)
        logs["recon_loss/all"] = all_mse

        weighted_sum = torch.tensor(0.0, device=device)
        wsum = 0.0

        # foreground
        for mk in mask_keys:
            cid = int(class_map.get(mk, 0))
            if cid <= 0:
                continue

            v = image_recon_error.get(f"label_{cid}_batch_mse", None)
            if v is None:
                if count_missing_as_zero:
                    v = torch.tensor(0.0, device=device)
                else:
                    continue
            else:
                v = v.to(device)

            logs[f"recon_loss/{mk}"] = v

            w = self._get_weight_for_key(
                mk=mk, cid=cid,
                mask_weights_cfg=mask_weights_cfg,
                class_weight_by_id=class_weight_by_id,
                default_w=1.0,
            )

            logs[f"recon_loss_weighted/{mk}"] = v * float(w)
            weighted_sum = weighted_sum + v * float(w)
            wsum += float(w)

        # ---- background ----
        if include_background:
            bg_mse = image_recon_error.get("label_0_batch_mse", None)
            if bg_mse is not None:
                bg_mse = bg_mse.to(device)
            else:
                bg_mse = torch.tensor(0.0, device=device)

            logs["recon_loss/background"] = bg_mse

            # bg weight（安全にfallback）
            if isinstance(mask_weights_cfg, dict):
                bg_w = float(mask_weights_cfg.get(background_key, 0.0))
            else:
                bg_w = 0.0

            logs["recon_loss_weighted/background"] = bg_mse * bg_w
            weighted_sum = weighted_sum + bg_mse * bg_w
            wsum += bg_w

        # normalize
        recon_loss_weighted = (weighted_sum / float(wsum)) if wsum > 0.0 else all_mse
        logs["recon_loss"] = recon_loss_weighted

        # # ===== DEBUG PRINT =====
        # do_print = debug and self._is_rank0() and (step is None or (debug_every > 0 and step % debug_every == 0))
        # if do_print:
        #     print("\n=== _aggregate_recon_loss_weighted DEBUG ===")
        #     print(f"step={step} wsum={wsum:.6f}")
        #     print(f"recon_loss/all={self._to_float(logs['recon_loss/all']):.6f}")
        #     print(f"weighted_sum={self._to_float(weighted_sum):.6f}")
        #     print(f"recon_loss={self._to_float(logs['recon_loss']):.6f}")

        #     for mk in mask_keys:
        #         k = f"recon_loss/{mk}"
        #         kw = f"recon_loss_weighted/{mk}"
        #         if k in logs:
        #             mse = self._to_float(logs[k])
        #             contrib = self._to_float(logs[kw]) if kw in logs else float("nan")
        #             w_est = (contrib / mse) if abs(mse) > 1e-12 else float("nan")
        #             print(f"  {mk:>12s}: mse={mse:.6f} w~={w_est:.6f} contrib={contrib:.6f}")

        #     if "recon_loss/background" in logs:
        #         mse = self._to_float(logs["recon_loss/background"])
        #         contrib = self._to_float(logs["recon_loss_weighted/background"])
        #         w_est = (contrib / mse) if abs(mse) > 1e-12 else float("nan")
        #         print(f"  {'background':>12s}: mse={mse:.6f} w~={w_est:.6f} contrib={contrib:.6f}")

        #     print("===========================================\n")

        return recon_loss_weighted, logs


    def prepare_masks_multiclass_dict(
        self,
        batch: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str], Dict[str, Any]]:
        """
        <mask_key> ごとに class_id を割り当て、多クラスラベルマップとして統合して返す。

        返り値:
        masks_dict[cam] = [B,T,1,H',W'] int64 (0=bg, 1..K=class)
        pad_dict[cam]   = [B,T] bool
        mask_keys       = class_id=1..K に対応する mask_key の順序リスト（bgは含めない）
        meta            = クラス名/ID/重み等のメタ情報（cam共通）
        """
        feat_keys = list(self.config.mask_features)
        if len(feat_keys) == 0:
            return {}, {}, [], {"classes": []}

        # cam ごとに feature keys を束ねる
        cam_to_feats: Dict[str, List[str]] = {}
        for k in feat_keys:
            if "/mask/" not in k:
                continue
            cam = k.split("/mask/")[0]
            cam_to_feats.setdefault(cam, []).append(k)

        if len(cam_to_feats) == 0:
            raise ValueError(f"No mask feature keys with '/mask/' found: {feat_keys}")

        # feat_keys から実際に出現しうる mask_key 集合
        feat_mask_keys_all = sorted({self.extract_mask_key(k) for k in feat_keys})

        # -----------------------------
        # mask_weights 対応（background を特別扱い）
        # -----------------------------
        raw_mask_weights = getattr(self.config, "mask_weights", None)
        bg_aliases = {"background", "bg"}

        bg_weight = 1.0
        mask_keys: List[str]
        mask_weights: Optional[Dict[str, float]]

        if isinstance(raw_mask_weights, dict) and len(raw_mask_weights) > 0:
            ordered = [(str(k), float(v)) for k, v in raw_mask_weights.items()]

            # background 重み
            for k, w in ordered:
                if k.lower() in bg_aliases:
                    bg_weight = float(w)
                    break
            self.mask_background_weight = float(bg_weight)

            # bg以外 & feat側に存在するキーだけ採用（順序はmask_weightsを尊重）
            primary = [k for k, _ in ordered if (k.lower() not in bg_aliases) and (k in feat_mask_keys_all)]
            rest = [k for k in feat_mask_keys_all if k not in primary]
            mask_keys = primary + rest

            tmp = {k: float(v) for k, v in ordered if k.lower() not in bg_aliases}
            tmp = {k: tmp[k] for k in tmp.keys() if k in feat_mask_keys_all}  # featに無いものは無視
            for k in rest:
                tmp.setdefault(k, 1.0)  # 未指定は1.0
            mask_weights = tmp
        else:
            mask_keys = feat_mask_keys_all
            mask_weights = None
            self.mask_background_weight = float(bg_weight)

        # -----------------------------
        # class_map を作る（bgは 0）
        # -----------------------------
        class_map = getattr(self, "mask_class_map", None)
        if class_map is None:
            class_map = self.build_mask_class_map(mask_keys)  # 1..K を割り当て
            self.mask_class_map = class_map

        # bg alias は常に 0
        class_map["background"] = 0
        class_map["bg"] = 0

        # class_id -> weight（bg含む）を作る
        id_to_weight: Dict[int, float] = {0: float(bg_weight)}
        if mask_weights is not None:
            for mk in mask_keys:
                cid = int(class_map.get(mk, 0))
                if cid > 0:
                    id_to_weight[cid] = float(mask_weights.get(mk, 1.0))
            self.mask_class_weight_by_id = {k: v for k, v in id_to_weight.items() if k != 0}
        else:
            # weight未指定なら fg=1.0
            for mk in mask_keys:
                cid = int(class_map.get(mk, 0))
                if cid > 0:
                    id_to_weight[cid] = 1.0
            self.mask_class_weight_by_id = None

        # id_to_name / name_to_id を整備（bgを正規名 "background" に寄せる）
        id_to_name: Dict[int, str] = {0: "background"}
        for mk in mask_keys:
            cid = int(class_map.get(mk, 0))
            if cid > 0:
                id_to_name[cid] = str(mk)

        name_to_id: Dict[str, int] = {v: k for k, v in id_to_name.items()}

        # class一覧（0..K の順）
        max_cid = max(id_to_name.keys()) if len(id_to_name) > 0 else 0
        classes = []
        for cid in range(max_cid + 1):
            nm = id_to_name.get(cid, f"unknown_{cid}")
            wt = float(id_to_weight.get(cid, 1.0 if cid != 0 else bg_weight))
            classes.append({"id": cid, "name": nm, "weight": wt})

        meta: Dict[str, Any] = {
            "classes": classes,                 # [{"id":0,"name":"background","weight":...}, ...]
            "id_to_name": id_to_name,           # {0:"background", 1:"robot", ...}
            "name_to_id": name_to_id,           # {"background":0, "robot":1, ...}
            "id_to_weight": id_to_weight,       # {0:bg_w, 1:w1, ...}
            "background_weight": float(bg_weight),
        }

        # -----------------------------
        # masks_dict / pad_dict を生成
        # -----------------------------
        masks_dict: Dict[str, Tensor] = {}
        pad_dict: Dict[str, Tensor] = {}

        for cam, keys in cam_to_feats.items():
            present = [k for k in keys if k in batch]
            if len(present) == 0:
                raise ValueError(f"All mask features for cam '{cam}' are missing from batch. expected={keys}")

            m0, B, T, H, W = self._to_bt1hw(batch[present[0]])
            label = torch.zeros((B, T, 1, H, W), dtype=torch.int64, device=m0.device)

            pad_key0 = f"{present[0]}_padding_mask"
            if pad_key0 in batch:
                pm = batch[pad_key0].bool()
                if pm.ndim == 1:
                    pm = pm.unsqueeze(1).expand(B, T)
            else:
                pm = torch.ones((B, T), dtype=torch.bool, device=m0.device)

            for k in present:
                mk = self.extract_mask_key(k)
                if mk.lower() in bg_aliases:
                    continue

                cid = int(class_map.get(mk, 0))
                if cid <= 0:
                    continue

                m, B2, T2, H2, W2 = self._to_bt1hw(batch[k])
                if (B2, T2, H2, W2) != (B, T, H, W):
                    raise ValueError(
                        f"Shape mismatch among masks for cam={cam}: base={(B,T,H,W)} vs {k}={(B2,T2,H2,W2)}"
                    )

                on = (m > 0)
                label = torch.where(on, torch.full_like(label, cid), label)

                pk = f"{k}_padding_mask"
                if pk in batch:
                    pm_k = batch[pk].bool()
                    if pm_k.ndim == 1:
                        pm_k = pm_k.unsqueeze(1).expand(B, T)
                    pm = pm & pm_k

            if getattr(self.config, "resize_imgs_with_padding", None) is not None:
                rh, rw = self.config.resize_imgs_with_padding
                label_flat = label.reshape(B * T, 1, H, W).float()
                label_flat = resize_with_pad_mask(label_flat, rw, rh)
                label = label_flat.reshape(B, T, 1, rh, rw).to(torch.int64)

            masks_dict[cam] = label
            pad_dict[cam] = pm

        return masks_dict, pad_dict, mask_keys, meta

    def prepare_state(self, batch: Dict[str, Tensor]) -> Tensor:
        """ロボット状態ベクトルをパディングして固定長に揃える。"""
        state = batch[OBS_STATE]
        # state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        # state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """行動ベクトルをパディングして固定長に揃える。"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    # ------------------------------------------------------------------
    #  ALOHA 互換：必要な場合のみ
    # ------------------------------------------------------------------
    def _pi_aloha_decode_state(self, state: Tensor) -> Tensor:
        # Flip some joints
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse gripper transform
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions: Tensor) -> Tensor:
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions: Tensor) -> Tensor:
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions
