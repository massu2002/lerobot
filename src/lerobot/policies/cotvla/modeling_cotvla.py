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
from typing import Deque, Dict, List, Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

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

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
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
            future_image_primary=images[2],
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
    def forward(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """トレーニング時の forward（損失計算）。

        戻り値:
            loss:   backward に使うスカラー Tensor
            loss_dict: ログ用に中間の loss テンソルをいくつか保持した dict
        """
        if getattr(self.config, "adapt_to_pi_aloha", False):
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            # 既存データのアクションを Aloha 互換の joint 表現に変換
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad", None)

        loss_dict: Dict[str, Tensor] = {}

        # CoTVLA 本体の forward
        _, loss_arm_action, _, image_recon_error = self.model.forward(
            image_primary=images[1],
            image_wrist=images[0],
            state=state,
            text_token=lang_tokens,
            text_attn=lang_masks,
            action_label=actions,
            mode="train",
            future_image_primary=images[2],
            future_image_wrist=images[3] if len(images) > 3 else None,
        )
        if image_recon_error is not None:
            img_recon_scalar = image_recon_error["batch_mse"]
            losses = loss_arm_action + self.config.img_recon_loss_weight * img_recon_scalar
        else:
            losses = loss_arm_action
        loss_dict["losses_after_forward"] = losses.clone()
        
        loss_dict["image_recon_error"] = image_recon_error["batch_mse"] if image_recon_error is not None else torch.tensor(0.0)
        loss_dict["action_loss"] = loss_arm_action

        # エピソード外パディングを無視
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad  # True = 有効ステップ
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # action 次元のパディングを除去
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # scalar loss
        loss = losses.mean()
        loss_dict["loss"] = loss

        return loss, loss_dict

    # ------------------------------------------------------------------
    #  Feature preparation helpers
    # ------------------------------------------------------------------
    def prepare_images(self, batch: Dict[str, Tensor]):
        """CoTVLA 用の画像前処理。

        - (B, T, C, H, W) の場合は **全フレーム T を使用**（CoTVLA 仕様）
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
