# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.utils.constants import OBS_IMAGES
from typing import Optional, Tuple


@PreTrainedConfig.register_subclass("cotvla")
@dataclass
class CoTVLAConfig(PreTrainedConfig):
    """
    CoT-VLA モデル用の設定クラス。
    SmolVLAConfig の構造をベースに、CoT-VLA の argparse 引数を中心に取り込んでいます。
    """
    # pretrain / finetune / evaluate
    phase: str = "pretrain"

    model_type: str = "cotvla"
    
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    # Shorter state and action vectors will be padded
    max_state_dim: int = 6
    max_action_dim: int = 6
    
    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # ======================
    # 1. トレーニング関連（モデルに紐づくもの）
    # ======================
    optimizer_lr: float = 1e-4
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4 # 0.05
    optimizer_grad_clip_norm: float = 10  # 1.0

    scheduler_warmup_steps: int = 1_000 # 5000
    scheduler_decay_steps: int = 40_000 # 156000
    scheduler_decay_lr: float = 2.5e-6 # 1.5e-5

    # ======================
    # 2. データ関連（モデルに依存するものだけ）
    # ======================
    empty_cameras: int = 0

    # ======================
    # 3. CoT-VLA モデル構造
    # ======================
    
    # Visual CoT 関連
    patch_size: int = 16
    obs_pred: bool = True  # Visual CoT を使うかどうか
    use_sam_mask_loss: bool = False  # SAMマスク領域だけ(／以外)の再構成Lossを「分けて」取得するか？

    # シーケンス / トークン関連
    pred_num: int = 1
    mask_l_obs_ratio: float = 0.0
    num_resampler_query: int = 9
    num_obs_token_per_image: int = 9

    # ViT / DiT
    vit_checkpoint_path: Optional[str] = "../checkpoints/mae/mae_pretrain_vit_base.pth"

    # GPT-2 / Transformer 側
    transformer_layers: int = 12
    hidden_dim: int = 1024
    transformer_heads: int = 12
    use_gpt2_pretrained: bool = True
    attn_implementation: str = "eager"

    # ======================
    # 4. アテンション / 条件付けフラグ
    # ======================

    # attention config
    atten_only_obs: bool = False
    attn_robot_proprio_state: bool = False
    atten_goal: int = 0
    atten_goal_state: bool = False

    # ======================
    # 5. 損失関数関連
    # =====================
    mask_keys : list[str] = field(default_factory=lambda: [])
    img_recon_loss_weight: float = 1.0

    def __post_init__(self):
        # PreTrainedConfig 側の初期化
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
    
    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
