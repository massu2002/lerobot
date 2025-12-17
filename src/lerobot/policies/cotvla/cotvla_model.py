import os
import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn, Tensor
import torch.nn.functional as F
import clip
import numpy as np
from lerobot.policies.cotvla.models.vit_mae import MaskedAutoencoderViT
from lerobot.policies.cotvla.models.perceiver_resampler import PerceiverResampler
from lerobot.policies.cotvla.models.gpt2 import GPT2Model
from transformers import CLIPTextModel, GPT2Config
from pdb import set_trace
import random 
from lerobot.policies.cotvla.action_model.action_model import  ActionModel, ActionModelFM
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        diff_log = torch.log(target+ 1e-6) - torch.log(pred+1e-6)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss

def generate_attention_mask(K, num_A, num_B, atten_goal, atten_goal_state,
                            atten_only_obs,
                            attn_robot_proprio_state,
                            mask_l_obs_ratio,
                            num_obs_token, n_action_steps):
    # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
    # num_A: text, state, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN+self.n_action_steps
    # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
    n_obs_steps = (num_A + num_B) * K
    attention_mask = torch.zeros((n_obs_steps, n_obs_steps))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')
        
        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = -float('inf')
        
        # if obs_token exists, action_pred_token should attend to it
        if num_obs_token > 0 and n_action_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+n_action_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
        if num_obs_token > 0 and atten_only_obs and n_action_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+n_action_steps] = -float('inf')
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+n_action_steps, start_index+2:start_index+num_A] = 0.0
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+n_action_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
            if attn_robot_proprio_state:
                attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+n_action_steps, start_index+1:start_index+2] = 0.0
            if mask_l_obs_ratio > 0:
                count = int(mask_l_obs_ratio * (num_obs_token))
                selected_numbers = np.random.choice(range(num_obs_token), size=count, replace=False)
                for num in selected_numbers:
                    attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+n_action_steps, start_index+num_A+num] = -float('inf')
        if num_obs_token > 0 and atten_goal:
            if i < K - atten_goal:
                pred_end_index = (i + atten_goal) * (num_A + num_B)
                if atten_goal_state:
                    attention_mask[start_index+num_A:start_index+num_A+num_obs_token,pred_end_index+1:pred_end_index+2] = 0.0

    return attention_mask

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

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


# =========================
# Text / State Encoder Initialization
# =========================
class MultiTokenProjector(nn.Module):
    """
    入力系列 [B, L_in, D_in] を L_out トークンに圧縮する学習可能なプロジェクタ
    （簡易Perceiver風構造）
    """
    def __init__(self, D_in: int, D_hidden: int, num_query: int = 1, num_heads: int = 8):
        super().__init__()
        self.num_query = num_query
        self.query = nn.Parameter(torch.randn(num_query, D_in))
        self.cross_attn = nn.MultiheadAttention(D_in, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Linear(D_in, D_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L_in, D_in]
        B, L_in, D_in = x.shape
        query = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, num_query, D_in]
        attn_out, _ = self.cross_attn(query, x, x)         # [B, num_query, D_in]
        return self.proj_out(attn_out)                     # [B, num_query, D_hidden]


class CoTVLA(nn.Module):
    """
    CoT-VLA 仕様に特化した版：

    - Visual CoT: 未来観測の再構成 (obs_pred=True)
    """

    def __init__(
        self,
        clip_device: torch.device,
        vit_checkpoint_path: str,
        *,
        n_obs_steps: int = 10,
        num_resampler_query: int = 9,
        num_obs_token_per_image: int = 10,
        n_action_steps: int = 10,
        resize_imgs_with_padding=(512, 512),
        patch_size: int = 16,
        transformer_layers: int = 12,
        hidden_dim: int = 384,
        transformer_heads: int = 12,
        phase: str = "pretrain",              # ["pretrain", "finetune", "evaluate"]
        pred_num: int = 1,                    # 未来フレーム数（サブゴール数）
        use_gpt2_pretrained: bool = False,
        attn_implementation: str | bool = False,  # "sdpa" など
        atten_only_obs: bool = False,
        attn_robot_proprio_state: bool = False,
        atten_goal: bool = False,
        atten_goal_state: bool = False,
        mask_l_obs_ratio: float = 0.0,
        action_dim: int = 6,
        text_vocab_size: int | None = None,   # GPT2 を自前初期化する場合の語彙サイズ
        action_model_type: str = "DiT-S",
        action_diffusion_steps: int = 100,
        action_noise_schedule: str = "squaredcos_cap_v2",
        obs_pred: bool = True,
    ):
        super().__init__()

        # --- 基本パラメータ ---
        self.device = clip_device
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.phase = phase
        self.obs_pred = obs_pred
        self.pred_num = pred_num
        assert self.phase in ["pretrain", "finetune", "evaluate"]

        # attention の制御フラグ（Hybrid Attention 用）
        self.atten_goal = atten_goal
        self.atten_goal_state = atten_goal_state
        self.atten_only_obs = atten_only_obs
        self.attn_robot_proprio_state = attn_robot_proprio_state
        self.mask_l_obs_ratio = mask_l_obs_ratio

        self.vit_checkpoint_path = vit_checkpoint_path

        # --- Text Projector (CLIP -> hidden_dim, トークン数調整付き) ---
        TEXT_INPUT_DIM = 512        # CLIP text feature 次元
        NUM_TEXT_TOKEN = 10         # 出力トークン数 (調節可能)

        
        self.num_text_token = NUM_TEXT_TOKEN
        self.text_projector = MultiTokenProjector(
            D_in=TEXT_INPUT_DIM,
            D_hidden=self.hidden_dim,
            num_query=NUM_TEXT_TOKEN,
        )

        # --- State Encoder (6次元 → hidden_dim, トークン数調整付き) ---
        STATE_INPUT_DIM = 6
        NUM_STATE_TOKEN = 1          # 出力トークン数 (調節可能)

        self.num_state_token = NUM_STATE_TOKEN
        self.state_encoder = MultiTokenProjector(
            D_in=STATE_INPUT_DIM,
            D_hidden=self.hidden_dim,
            num_query=NUM_STATE_TOKEN,
            num_heads=1,
        )

        # --- Vision Encoder (frozen MAE) ---
        self.vision_encoder = MaskedAutoencoderViT(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.RESAMPLER_hidden_dim = 768
        self.NUM_RESAMPLER_QUERY = num_resampler_query
        self.perceiver_resampler = PerceiverResampler(
            dim=self.RESAMPLER_hidden_dim,
            num_latents=self.NUM_RESAMPLER_QUERY,
            depth=3,
        )
        self.image_primary_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_primary_projector = nn.Linear(768, self.hidden_dim)
        self.image_wrist_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_wrist_projector = nn.Linear(768, self.hidden_dim)

        # --- Obs Tokens（Visual CoT Query）---
        self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
        self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 2
        self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.hidden_dim))
        
        # --- Action 設定 ---
        self.num_action_tokens_total = self.n_action_steps * self.action_dim # 予測アクショントークン数（連続値なので action_dim 倍）
        self.action_model = ActionModel(
            token_size=self.hidden_dim,
            model_type=action_model_type, 
            in_channels=self.action_dim,
            future_action_window_size=self.n_action_steps,
            past_action_window_size=0,
            diffusion_steps=action_diffusion_steps,
            noise_schedule=action_noise_schedule,
        )

        # --- Causal Transformer (GPT2 backbone) ---
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dim)

        # 1) Aトークン（条件側）の総数
        # text: NUM_TEXT_TOKEN
        # state: NUM_STATE_TOKEN
        # image: primary Q + wrist Q = 2 * NUM_RESAMPLER_QUERY
        num_text_token  = self.num_text_token
        num_state_token = self.num_state_token
        num_image_token = self.NUM_RESAMPLER_QUERY * 2

        num_A = num_text_token + num_state_token + num_image_token

        # 2) Bトークン（予測側＝obs + action）
        if self.obs_pred:
            this_num_obs_token = self.NUM_OBS_TOKEN
        else:
            this_num_obs_token = 0  # obs トークンを使わない
        # num_B = this_num_obs_token + self.num_action_tokens_total
        num_B = this_num_obs_token + 0

        # 3) attention mask 生成
        self.attention_mask = nn.Parameter(
            generate_attention_mask(
                K=self.n_obs_steps,
                num_A=num_A,
                num_B=num_B,
                atten_goal=self.atten_goal,
                atten_goal_state=self.atten_goal_state,
                atten_only_obs=self.atten_only_obs,
                attn_robot_proprio_state=self.attn_robot_proprio_state,
                mask_l_obs_ratio=self.mask_l_obs_ratio,
                num_obs_token=this_num_obs_token,
                n_action_steps=self.n_action_steps,
            ),
            requires_grad=False,
        )

        self.transformer_backbone_position_embedding = nn.Parameter(
            torch.zeros(1, self.n_obs_steps, 1, self.hidden_dim), requires_grad=True
        )

        # GPT2 backbone 構築
        if not use_gpt2_pretrained:
            gpt2_config = GPT2Config()
            if text_vocab_size is None:
                text_vocab_size = 32000
            gpt2_config.hidden_size = self.hidden_dim
            gpt2_config.n_layer = transformer_layers
            gpt2_config.vocab_size = text_vocab_size
            gpt2_config.n_head = transformer_heads
            self.transformer_backbone = GPT2Model(gpt2_config)
            self.attn_implementation = attn_implementation
            self.vocab_size_text = text_vocab_size
        else:
            self.transformer_backbone = GPT2Model.from_pretrained("gpt2-medium")
            self.attn_implementation = getattr(
                self.transformer_backbone.config,
                "_attn_implementation",
                attn_implementation,
            )
            self.vocab_size_text = self.transformer_backbone.config.vocab_size

        # --- Visual CoT: Obs Prediction Head (MAE-style) ---
        self.IMAGE_DECODER_hidden_dim = self.hidden_dim
        H, W = resize_imgs_with_padding
        self.PATCH_SIZE = patch_size
        self.NUM_MASK_TOKEN = int(H * W / patch_size / patch_size) * self.pred_num

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.IMAGE_DECODER_hidden_dim))
        self.image_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN,
                self.IMAGE_DECODER_hidden_dim,
            ),
            requires_grad=False,
        )
        self.image_decoder = nn.Sequential(
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
        )
        self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, self.PATCH_SIZE**2 * 3)

        # --- 初期化 & 凍結 ---
        self.initialize_weights()

        # vision encoder をロードして凍結
        vit_checkpoint = torch.load(self.vit_checkpoint_path, map_location="cpu")
        _ = self.vision_encoder.load_state_dict(vit_checkpoint["model"], strict=False)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        self.vision_encoder.eval()

        # CLIP テキストエンコーダをロードして凍結
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip_text_model.parameters():
            p.requires_grad = False

        # --- Visual CoT モジュールの凍結（obs_pred=False の場合） ---
        if not self.obs_pred:
            visual_cot_modules = [
                self.image_decoder_obs_pred_projector,
                self.image_decoder,
                self.image_decoder_norm,
                self.image_decoder_pred,
            ]
            for m in visual_cot_modules:
                for p in m.parameters():
                    p.requires_grad = False

            # mask_token と positional embedding も学習停止
            self.mask_token.requires_grad = False
            self.image_decoder_position_embedding.requires_grad = False
            self.obs_tokens.requires_grad = False

        # 残りの初期化
        self._init_model_type()
        self.use_gradient_checkpointing = False

    # ----- init helpers -----
    def initialize_weights(self):
        # Visual CoT デコーダの pos embed を sin-cos で初期化
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(
            self.IMAGE_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_IMAGE ** 0.5), cls_token=False
        )
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(
            self.IMAGE_DECODER_hidden_dim, int(self.NUM_MASK_TOKEN ** 0.5), cls_token=False
        )
        image_decoder_position_embedding = np.concatenate(
            (image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0
        )
        self.image_decoder_position_embedding.data.copy_(
            torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)

        torch.nn.init.normal_(self.transformer_backbone_position_embedding, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_model_type(self):
        self.vision_encoder_type = next(self.vision_encoder.parameters()).type()
        self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).type()
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).type()
        
    def _patchify_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, C, H, W]
        return: [N, L, patch_dim]  (L = H*W / P^2, patch_dim = P*P*C)
        """
        P = self.PATCH_SIZE
        N, C, H, W = imgs.shape
        assert H % P == 0 and W % P == 0, "H, W は patch_size の倍数である必要があります"

        h = H // P
        w = W // P
        x = imgs.reshape(N, C, h, P, w, P)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [N, h, w, P, P, C]
        x = x.view(N, h * w, P * P * C)
        return x  # [N, L, patch_dim]
    
    def _unpatchify_images(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: [N, L, patch_dim] 
            - L = (H / P) * (W / P)
            - patch_dim = P * P * C
        return: [N, C, H, W]
        """
        P = self.PATCH_SIZE
        N, L, patch_dim = patches.shape
        C = patch_dim // (P * P)

        # もとの画像のパッチ配置 (h, w)
        h = w = int(L ** 0.5)
        assert h * w == L, f"L={L} は正方パッチ数に対応している必要があります"

        # [N, h, w, P, P, C] に戻す
        x = patches.view(N, h, w, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [N, C, h, P, w, P]
        imgs = x.view(N, C, h * P, w * P)
        return imgs  # [N, C, H, W]


    def _build_image_recon_target(
        self,
        future_primary: torch.Tensor,  # [B, pred_num, C, H, W]
        future_wrist:   torch.Tensor,  # [B, pred_num, C, H, W]
    ) -> torch.Tensor:
        """
        公式 CoT-VLA 寄せの再構成ターゲット生成:
        - future_primary, future_wrist: [B, pred_num, C, H, W]
        - 出力: [B, 2, pred_num, L, patch_dim]
        """
        B, Pn, C, H, W = future_primary.shape
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        prim_imgs  = future_primary.to(device=device, dtype=dtype).view(B * Pn, C, H, W)
        wrist_imgs = future_wrist.to(device=device, dtype=dtype).view(B * Pn, C, H, W)

        prim_patches  = self._patchify_images(prim_imgs)   # [B*Pn, L, patch_dim]
        wrist_patches = self._patchify_images(wrist_imgs)  # [B*Pn, L, patch_dim]

        N, L, patch_dim = prim_patches.shape
        prim_patches  = prim_patches.view(B, Pn, L, patch_dim)
        wrist_patches = wrist_patches.view(B, Pn, L, patch_dim)

        target = torch.stack([prim_patches, wrist_patches], dim=1)  # [B, 2, pred_num, L, patch_dim]
        return target

    # ----- forward -----
    def forward(
        self, 
        image_primary,          # [B, S_obs, C, H, W]
        image_wrist,            # [B, S_obs, C, H, W]
        state,                  # [B, S_obs, 6]
        text_token,             # [B, L]
        text_attn,              # [B, L]
        action_label=None,      # [B, T_action, action_dim] (train 時のみ)
        mode: str = "train",
        future_image_primary: torch.Tensor | None = None,  # [B, pred_num, C, H, W] （公式の s_{t+n} 用）
        future_image_wrist:  torch.Tensor | None = None,   # 同上
    ):
        """
        CoT-VLA 仕様寄せ版 forward

        入力:
            image_primary      : 現在〜過去の観測画像 (メインカメラ) [B, S_obs, C, H, W]
            image_wrist        : 現在〜過去の観測画像 (リストカメラ)   [B, S_obs, C, H, W]
            state              : ロボット状態 [B, S_obs, 6]
            text_token         : CLIP テキストトークン [B, L]
            action_label       : 連続アクション [B, T_action, action_dim]
            future_image_*     : subgoal 用の将来画像 (s_{t+n}) [B, pred_num, C, H, W]
                                → 論文の (l, s_t, s_{t+n}, a_t...a_{t+m}) の s_{t+n} に対応
        戻り値:
            arm_pred_action : [B, n_action_steps, action_dim] 連続アクション
            loss_arm_action : Cross Entropy loss (train 時のみ, それ以外は None)
            image_pred      : 未来画像パッチ予測（Visual CoT）
            image_recon_error : dict or None
        """
        # =========================
        # 0. 形状の整理
        # =========================
        B, S_obs, _ = state.shape

        image_primary_all = image_primary      # [B, S_obs, C, H, W]
        image_wrist_all   = image_wrist        # [B, S_obs, C, H, W]
        state_all         = state              # [B, S_obs, 6]
        S = S_obs

        # =========================
        # 1. Attention Mask（必要なら再生成）
        # =========================
        if self.phase == "pretrain":
            if self.obs_pred:
                this_num_obs_token = self.NUM_OBS_TOKEN
            else:
                this_num_obs_token = 0  # obs トークンを使わない
            num_text_token  = self.num_text_token
            num_state_token = self.num_state_token
            num_image_token = self.NUM_RESAMPLER_QUERY * 2
            num_A = num_text_token + num_state_token + num_image_token

            # 今は action 側トークンは 0 個なので num_B = this_num_obs_token
            num_B = this_num_obs_token
            K = S  # 実際の観測ステップ数

            new_mask = generate_attention_mask(
                K=K,
                num_A=num_A,
                num_B=num_B,
                atten_goal=self.atten_goal,
                atten_goal_state=self.atten_goal_state,
                atten_only_obs=self.atten_only_obs,
                attn_robot_proprio_state=self.attn_robot_proprio_state,
                mask_l_obs_ratio=self.mask_l_obs_ratio,
                num_obs_token=this_num_obs_token,
                n_action_steps=self.n_action_steps,
            ).to(self.device)

            if (
                not hasattr(self, "attention_mask")
                or self.attention_mask is None
                or self.attention_mask.shape != new_mask.shape
            ):
                self.attention_mask = nn.Parameter(new_mask, requires_grad=False)

        image_pred = None
        arm_pred_action = None
        loss_arm_action = None
        image_recon_error = None

        # =========================
        # 2. Text Embedding (token-wise)
        # =========================
        with torch.no_grad():
            text_outputs = self.clip_text_model(
                input_ids=text_token,
                attention_mask=text_attn,
                output_hidden_states=False,
            )
            text_feature_token = text_outputs.last_hidden_state  # [B, L, D_text]

        # 必要に応じて projector で hidden_dim に合わせる
        text_token_embedding = self.text_projector(text_feature_token)       # [B, NUM_TEXT_TOKEN, D]
        text_embedding = text_token_embedding.unsqueeze(1).expand(B, S, -1, -1)  # [B, S, NUM_TEXT_TOKEN, D]
        
        # =========================
        # 3. State Embedding
        # =========================
        # state_all: [B, S, 6]
        state_flat = state_all.flatten(0, 1)          # [B*S, 6]

        # MultiTokenProjector は [B, L_in, D_in] を期待するので
        # L_in=1 の「疑似系列」にして渡す
        state_seq = state_flat[:, :self.action_dim].unsqueeze(1)  # [B*S, 1, 6]

        # state_encoder: MultiTokenProjector(D_in=6, D_hidden=hidden_dim, num_query=NUM_STATE_TOKEN)
        state_feature = self.state_encoder(state_seq)  # [B*S, NUM_STATE_TOKEN, hidden_dim]

        # 元の [B, S] に戻して 4次元テンソルへ
        state_embedding = state_feature.view(
            B,
            S,
            self.num_state_token,
            self.hidden_dim,
        )  # [B, S, NUM_STATE_TOKEN, D]

        # =========================
        # 4. Vision Embedding（MAE パッチ → Perceiver で数トークンに圧縮）
        # =========================
        vision_device = next(self.vision_encoder.parameters()).device
        image_primary_all = image_primary_all.to(vision_device, non_blocking=True)
        image_wrist_all   = image_wrist_all.to(vision_device, non_blocking=True)

        if image_primary_all.type() != self.vision_encoder_type:
            image_primary_all = image_primary_all.type(self.vision_encoder_type)
            image_wrist_all   = image_wrist_all.type(self.vision_encoder_type)

        with torch.no_grad():
            # MAE encoder で特徴抽出（CLS + patch）
            # [B*S, 1+L, 768]
            image_primary_feature, _, _ = self.vision_encoder.forward_encoder(
                image_primary_all.flatten(0, 1), mask_ratio=0.0
            )
            image_wrist_feature,  _, _ = self.vision_encoder.forward_encoder(
                image_wrist_all.flatten(0, 1), mask_ratio=0.0
            )

        # モデル側 dtype / device に合わせる
        model_device = next(self.parameters()).device
        model_dtype  = next(self.parameters()).dtype
        image_primary_feature = image_primary_feature.to(device=model_device, dtype=model_dtype, non_blocking=True)
        image_wrist_feature   = image_wrist_feature.to(device=model_device, dtype=model_dtype, non_blocking=True)

        # CLS と patch に分割（必要なら CLS はあとで別用途に使えるように保持）
        image_primary_cls_token = image_primary_feature[:, :1, :]    # [B*S, 1, 768]
        image_wrist_cls_token   = image_wrist_feature[:, :1, :]      # [B*S, 1, 768]
        image_primary_patches   = image_primary_feature[:, 1:, :]    # [B*S, L, 768]
        image_wrist_patches     = image_wrist_feature[:, 1:, :]      # [B*S, L, 768]

        # Perceiver Resampler と projector を正しい device に
        self.perceiver_resampler          = self.perceiver_resampler.to(model_device)
        self.image_primary_projector      = self.image_primary_projector.to(model_device)
        self.image_wrist_projector        = self.image_wrist_projector.to(model_device)
        self.cls_token_primary_projector  = self.cls_token_primary_projector.to(model_device)
        self.cls_token_wrist_projector    = self.cls_token_wrist_projector.to(model_device)

        # パッチ列 → Perceiver Resampler で少数トークンに圧縮
        # 入力: [B*S, L, D] → unsqueeze(1).unsqueeze(1) → [B*S, 1, 1, L, D]
        image_primary_latent = self.perceiver_resampler(
            image_primary_patches.reshape(B * S, -1, self.RESAMPLER_hidden_dim)
                .unsqueeze(1).unsqueeze(1)
        )   # [B*S, 1, 1, Q, D]

        image_wrist_latent = self.perceiver_resampler(
            image_wrist_patches.reshape(B * S, -1, self.RESAMPLER_hidden_dim)
                .unsqueeze(1).unsqueeze(1)
        )   # [B*S, 1, 1, Q, D]

        # latent を project → [B*S, Q, D] → [B, S, Q, D]
        image_primary_embedding = self.image_primary_projector(
            image_primary_latent.flatten(0, 2)         # [B*S, Q, D_resampler]
        ).view(B, S, self.NUM_RESAMPLER_QUERY, self.hidden_dim)

        image_wrist_embedding = self.image_wrist_projector(
            image_wrist_latent.flatten(0, 2)
        ).view(B, S, self.NUM_RESAMPLER_QUERY, self.hidden_dim)

        # primary / wrist の Perceiver トークンを並べて「画像トークン列」にする
        # → [B, S, 2*Q, D]
        image_embedding = torch.cat(
            (image_primary_embedding, image_wrist_embedding),
            dim=2,
        )  # [B, S, 2 * NUM_RESAMPLER_QUERY, D]

        # =========================
        # 5. マルチモーダル埋め込みを concat（text, state, image の3種類のトークン）
        # =========================
        embeddings = torch.cat(
            (text_embedding, state_embedding, image_embedding),
            dim=2,
        )  # [B, S, num_text_token + num_state_token + num_image_token, D]
        pred_token_start_idx = embeddings.shape[2]

        # =========================
        # 6. Visual CoT 用の追加トークンを後ろに付ける
        # =========================
        if self.obs_pred:
            # Visual CoT query tokens
            obs_tokens = self.obs_tokens.repeat(B, S, 1, 1)  # [B, S, NUM_OBS_TOKEN, D]

            # 全トークン列: [テキスト, 状態, 画像, obs]
            transformer_input = torch.cat(
                [embeddings, obs_tokens],
                dim=2,
            )  # [B, S, T_total, D]
        else:
            # obs_pred=False のときは obs_tokens を一切入れない
            transformer_input = embeddings  # [B, S, T_total(=num_A), D]
        
        # 位置埋め込みを足す（時間ステップ S 方向だけ）
        transformer_input = (
            transformer_input
            + self.transformer_backbone_position_embedding.repeat(
                B, 1, transformer_input.shape[2], 1
            )
        )

        # GPT2 への入力形状に reshape
        transformer_input = transformer_input.flatten(1, 2)  # [B, S*T_total, D]

        # =========================
        # 6. GPT2 Backbone forward（元の run_backbone ロジックはほぼそのまま）
        # =========================
        if transformer_input.type() != self.transformer_backbone_type:
            transformer_input = transformer_input.type(self.transformer_backbone_type)
        transformer_input = transformer_input.to(
            device=next(self.transformer_backbone.parameters()).device,
            dtype=next(self.transformer_backbone.parameters()).dtype,
            non_blocking=True,
        )

        use_ckpt = self.use_gradient_checkpointing

        def run_backbone(x: torch.Tensor) -> torch.Tensor:
            """
            x: [B, S*T_total, D]
            return: last_hidden_state 相当 [B, S*T_total, D]
            """
            x = self.embedding_layer_norm(x)

            if self.attn_implementation == "sdpa":
                mask_4d = (
                    self.attention_mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(x.shape[0], -1, -1, -1)
                    .contiguous()
                )
                outputs = self.transformer_backbone(
                    inputs_embeds=x,
                    attention_mask=mask_4d,
                )
            else:
                outputs = self.transformer_backbone(
                    inputs_embeds=x,
                    attention_mask=self.attention_mask,
                )

            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs

        if self.transformer_backbone_type == "torch.BFloat16Tensor":
            with autocast(dtype=torch.bfloat16):
                if use_ckpt:
                    last_hidden = checkpoint(run_backbone, transformer_input)
                else:
                    last_hidden = run_backbone(transformer_input)
        else:
            if use_ckpt:
                last_hidden = checkpoint(run_backbone, transformer_input)
            else:
                last_hidden = run_backbone(transformer_input)

        transformer_output = last_hidden.view(
            B, S, -1, self.hidden_dim
        )  # [B, S, T_total, D]
        
        # =========================
        # 7. Visual CoT: 未来画像のパッチ再構成 & 再構成誤差
        # =========================
        if self.obs_pred and (mode in ["train", "evaluate"]):
            # 1) 観測ステップは「最後の 1 ステップだけ」を使う
            # transformer_output: [B, S, T, D]
            obs_pred_feature = transformer_output[
                :, -1:,  # ← ":" から "-1:" に変更済み
                pred_token_start_idx : pred_token_start_idx + self.NUM_OBS_TOKEN,
                :
            ]  # [B, 1, NUM_OBS_TOKEN, D]

            B, S_eff, _, D = obs_pred_feature.shape  # S_eff = 1

            # [B*S_eff*NUM_OBS_TOKEN, D]
            obs_pred_embedding = self.image_decoder_obs_pred_projector(
                obs_pred_feature.reshape(-1, D)
            )

            # 画像ごとにグルーピング
            obs_pred_embedding = obs_pred_embedding.view(
                B * S_eff * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE),
                self.NUM_OBS_TOKEN_PER_IMAGE,
                self.IMAGE_DECODER_hidden_dim,
            )

            # mask token を複製
            num_img = self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE  # = 2 (primary & wrist)
            mask_tokens = self.mask_token.repeat(
                B * S_eff * num_img,
                self.NUM_MASK_TOKEN,
                1,
            )

            image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1)
            image_decoder_input = image_decoder_input + self.image_decoder_position_embedding
            image_decoder_output = self.image_decoder(image_decoder_input)
            image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN :, :]

            image_pred_feature = self.image_decoder_norm(
                image_pred_feature.reshape(-1, self.IMAGE_DECODER_hidden_dim)
            )
            # [B*S_eff*2*pred_num*N_mask, patch_dim]
            image_pred = self.image_decoder_pred(image_pred_feature)

            # 2) 「バッチ＝B」に揃えて view し直す
            n_mask_per_frame = self.NUM_MASK_TOKEN // self.pred_num
            patch_dim = image_pred.shape[-1]

            image_pred = image_pred.view(
                B * S_eff,        # = B
                num_img,          # = 2 (primary & wrist)
                self.pred_num,    # 未来フレーム数
                n_mask_per_frame, # N_mask
                patch_dim,        # patch_dim (= P^2 * 3)
            )  # [B, 2, pred_num, N_mask, patch_dim]

            # 3) パッチから再構成画像 (Ŝ_{t+n}) を作る
            # [B, 2, pred_num, N_mask, patch_dim] → [B*2*pred_num, N_mask, patch_dim]
            image_pred_flat = image_pred.view(
                B * num_img * self.pred_num,
                n_mask_per_frame,
                patch_dim,
            )

            # unpatchify: [B*2*pred_num, N_mask, patch_dim] -> [B*2*pred_num, 3, H, W]
            # （_patchify_images の逆関数として実装しておく）
            recon_imgs = self._unpatchify_images(image_pred_flat)  # [B*2*pred_num, 3, H, W]

            # [B, 2, pred_num, 3, H, W] に戻す
            B2, C, H, W = recon_imgs.shape
            recon_imgs = recon_imgs.view(
                B,
                num_img,          # 2 (primary / wrist)
                self.pred_num,
                C,
                H,
                W,
            )  # [B, 2, pred_num, 3, H, W]

            # ---- 再構成誤差の計算（future_image_* が与えられている場合）----
            if mode in ["train", "evaluate"]:

                image_recon_target = self._build_image_recon_target(
                    future_image_primary,  # [B, pred_num, C, H, W]
                    future_image_wrist,    # [B, pred_num, C, H, W]
                )  # [B, 2, pred_num, N_mask, patch_dim]

                target = image_recon_target.to(image_pred.device, image_pred.dtype)

                # 4) ここで 0 次元が一致する（どちらも B）ことを確認
                assert image_pred.shape[0] == target.shape[0], \
                    f"image_pred.shape[0]={image_pred.shape[0]}, target.shape[0]={target.shape[0]}"

                diff = image_pred - target

                patch_mse = (diff ** 2).mean(dim=-1)          # [B, 2, pred_num, N_mask]
                sample_mse = patch_mse.mean(dim=(1, 2, 3))    # [B]
                batch_mse = sample_mse.mean()                 # scalar

                image_recon_error = {
                    "patch_mse": patch_mse,
                    "sample_mse": sample_mse,
                    "batch_mse": batch_mse,
                    "reconstructed_images": recon_imgs,  # [B, 2, pred_num, 3, H, W]
                }

        # =========================
        # 8. Action Prediction: Diffusion ActionModel + Visual CoT 条件
        # =========================

        loss_arm_action = None
        arm_pred_action = None

        if self.n_action_steps > 0:

            if self.obs_pred:
                # ---- 8-1. 1回目 GPT 出力から「未来画像トークン (obs)」を取り出す ----
                future_obs_tokens = transformer_output[
                    :, -1,  # 最後の時間ステップ
                    pred_token_start_idx : pred_token_start_idx + self.NUM_OBS_TOKEN,
                    :
                ]  # [B, NUM_OBS_TOKEN, D]

                # ---- 8-2. text / state / image のコンテキストを集約 ----
                text_ctx_tokens  = text_embedding[:, -1, :, :]   # [B, NUM_TEXT_TOKEN,  D]
                state_ctx_tokens = state_embedding[:, -1, :, :]  # [B, NUM_STATE_TOKEN, D]
                image_ctx_tokens = image_embedding[:, -1, :, :]  # [B, NUM_IMAGE_TOKEN, D]

                # ---- 8-3. 「context + 予測画像トークン」を 2nd-pass GPT 入力列にする ----
                cot_tokens = torch.cat(
                    [text_ctx_tokens, state_ctx_tokens, image_ctx_tokens, future_obs_tokens],
                    dim=1,
                )  # [B, L_cot, D]

                llm_device = next(self.transformer_backbone.parameters()).device
                llm_dtype  = next(self.transformer_backbone.parameters()).dtype
                cot_tokens = cot_tokens.to(device=llm_device, dtype=llm_dtype)

                cot_tokens = self.embedding_layer_norm(cot_tokens)  # [B, L_cot, D]

                if self.transformer_backbone_type == "torch.BFloat16Tensor":
                    with autocast(dtype=torch.bfloat16):
                        cot_outputs = self.transformer_backbone(inputs_embeds=cot_tokens)
                else:
                    cot_outputs = self.transformer_backbone(inputs_embeds=cot_tokens)

                if hasattr(cot_outputs, "last_hidden_state"):
                    cot_hidden = cot_outputs.last_hidden_state      # [B, L_cot, D]
                else:
                    cot_hidden = cot_outputs                        # [B, L_cot, D]

                # 最後のトークンを cond_z に
                cond_z = cot_hidden[:, -1, :]  # [B, D]

            else:
                #  1回目 GPT 出力だけから cond_z を作る
                # transformer_output: [B, S, T_total, D]
                # 最後の観測ステップ S-1 のトークン列を平均 pooling
                last_step_hidden = transformer_output[:, -1, :, :]   # [B, T_total, D]
                cond_z = last_step_hidden[:, -1, :]                  # [B, D]

            # -------------------------------------------------
            # 8-x. 学習時: ActionModel (DiT) の拡散損失
            # -------------------------------------------------
            if (action_label is not None) and (mode == "train"):
                B_act, T_action, A_dim = action_label.shape
                assert A_dim == self.action_dim
                assert T_action >= self.n_action_steps

                x0 = action_label[:, : self.n_action_steps, :]  # [B, T, C]
                loss_arm_action = self.action_model.loss(x0, cond_z)

            # -------------------------------------------------
            # 8-x. 評価 / 推論時: Diffusion サンプリング
            # -------------------------------------------------
            if (mode in ["evaluate", "inference"]) or (action_label is None):
                with torch.no_grad():
                    arm_pred_action = self.action_model.sample(
                        z=cond_z,
                        n_action_steps=self.n_action_steps,
                        action_dim=self.action_dim,
                        use_ddim=True,
                        ddim_step=10,
                        device=cond_z.device,
                    )

        return arm_pred_action, loss_arm_action, image_pred, image_recon_error

def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper