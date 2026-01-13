"""
action_model.py
"""

from .models import DiT
from ..action_model import create_diffusion
from . import gaussian_diffusion as gd
from .respace import FMDiffusion, space_timesteps
import torch
from torch import nn
from typing import Optional

# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

# Create ActionModel
class ActionModel(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 100,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)
        
        z_seq = z.unsqueeze(1)                         # [B, 1, D]

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z_seq)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(timestep_respacing = "ddim"+str(ddim_step), 
                                               noise_schedule = self.noise_schedule,
                                               diffusion_steps = self.diffusion_steps, 
                                               sigma_small = True, 
                                               learn_sigma = False
                                               )
        return self.ddim_diffusion

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        n_action_steps: int,
        action_dim: int,
        use_ddim: bool = True,
        ddim_step: int = 10,
        device=None,
        x_T: torch.Tensor | None = None,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            z: condition embedding [B, D]
            n_action_steps: T
            action_dim: C
            use_ddim: DDIM sampling if True else ancestral sampling
            ddim_step: number of DDIM steps
            device: sampling device (default: z.device)
            x_T: optional initial noise [B, T, C]
            clip_denoised: whether to clip predicted x0 (if diffusion supports)

        Returns:
            x_0: sampled action tokens [B, T, C]
        """
        if device is None:
            device = z.device

        B = z.shape[0]
        z_seq = z.to(device).unsqueeze(1)  # [B, 1, D]

        # initial noise
        if x_T is None:
            x = torch.randn(B, n_action_steps, action_dim, device=device)
        else:
            x = x_T.to(device)
            assert x.shape == (B, n_action_steps, action_dim), f"x_T.shape={x.shape}"

        # pick diffusion object
        diff = self.diffusion
        if use_ddim:
            if (self.ddim_diffusion is None) or (getattr(self.ddim_diffusion, "timestep_respacing", None) != "ddim" + str(ddim_step)):
                diff = self.create_ddim(ddim_step=ddim_step)
            else:
                diff = self.ddim_diffusion

        # time indices (descending)
        # create_diffusion("ddimK") usually changes num_timesteps to K.
        timesteps = list(range(diff.num_timesteps))[::-1]

        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # predict eps (noise)
            eps = self.net(x, t_batch, z_seq)  # [B, T, C]

            # --- Update rule ---
            # Prefer diffusion helper APIs if available. Many diffusion libs provide:
            #  - p_sample(...) for ancestral
            #  - ddim_sample(...) for DDIM
            if use_ddim and hasattr(diff, "ddim_sample"):
                out = diff.ddim_sample(
                    model=lambda _x, _t, **kw: self.net(_x, _t, z_seq),
                    x=x,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                )
                # common conventions: dict with "sample" key
                x = out["sample"] if isinstance(out, dict) and "sample" in out else out
            elif (not use_ddim) and hasattr(diff, "p_sample"):
                out = diff.p_sample(
                    model=lambda _x, _t, **kw: self.net(_x, _t, z_seq),
                    x=x,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                )
                x = out["sample"] if isinstance(out, dict) and "sample" in out else out
            else:
                # Fallback: implement DDIM-like step using predicted eps
                # (This branch is only used if your diffusion object lacks helpers.)
                # We need alphas_cumprod etc. Many libs expose them as arrays/tensors.
                alphas_cumprod = torch.as_tensor(diff.alphas_cumprod, device=device, dtype=x.dtype)
                if t == 0:
                    alpha_bar_prev = torch.ones((), device=device, dtype=x.dtype)
                else:
                    alpha_bar_prev = alphas_cumprod[t - 1]
                alpha_bar = alphas_cumprod[t]

                # x0 estimate
                x0 = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
                if clip_denoised:
                    x0 = x0.clamp(-1.0, 1.0)

                if use_ddim:
                    # deterministic DDIM (eta=0)
                    x = alpha_bar_prev.sqrt() * x0 + (1 - alpha_bar_prev).sqrt() * eps
                else:
                    # ancestral-ish: add noise (simple)
                    noise = torch.randn_like(x) if t > 0 else 0.0
                    x = alpha_bar_prev.sqrt() * x0 + (1 - alpha_bar_prev).sqrt() * eps + 0.0 * noise

        return x
    
    
class ActionModelFM(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 10,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device).float()
        timestep /= self.diffusion.num_timesteps

        # sample x_t from x
        # x_t = self.diffusion.q_sample(x, timestep, noise)
        x_t = timestep.view(-1, 1, 1) * x + (1 - timestep.view(-1, 1, 1)) * noise

        # predict noise from x_t
        ut = self.net(x_t, timestep, z)

        assert ut.shape == noise.shape == x.shape
        # Compute L2 loss
        # loss = ((noise_pred - noise) ** 2).mean()
        loss = ((ut - (x - noise)) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        timestep_respacing = "ddim"+str(ddim_step)
        noise_schedule = self.noise_schedule
        diffusion_steps = self.diffusion_steps 
        sigma_small = True
        learn_sigma = False
        
        
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        loss_type = gd.LossType.MSE
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        self.ddim_diffusion = FMDiffusion(
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type
            # rescale_timesteps=rescale_timesteps,
        )
        return self.ddim_diffusion
    
    # --------------------------------------------------
    # (B, n_action_steps, action_dim) をサンプリングする関数
    # --------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        n_action_steps: int,
        action_dim: int,
        use_ddim: bool = False,
        ddim_step: int = 10,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        条件ベクトル z から、(B, n_action_steps, action_dim) の行動列をサンプリングする。

        Parameters
        ----------
        z : torch.Tensor
            条件テンソル。形状 [B, D] を想定。
        n_action_steps : int
            予測したいステップ数 (T)。
        action_dim : int
            各ステップの行動次元数 (C)。
        use_ddim : bool, default False
            True の場合は DDIM サンプラーを使用。
        ddim_step : int, default 10
            DDIM のステップ数。use_ddim=True のときのみ使用。
        device : torch.device, optional
            サンプリングに使う device。None の場合は z.device を使用。

        Returns
        -------
        actions : torch.Tensor
            形状 [B, n_action_steps, action_dim] の行動列。
        """
        if device is None:
            device = z.device

        B = z.size(0)
        shape = (B, n_action_steps, action_dim)

        # 条件をネットワークのフォーマットに合わせる
        # loss() と同様に [B, 1, D] にして渡す
        z = z.to(device)
        z_seq = z.unsqueeze(1)  # [B, 1, D]

        # 使用する拡散オブジェクトを選択
        if use_ddim:
            if self.ddim_diffusion is None:
                self.create_ddim(ddim_step=ddim_step)
            diffusion = self.ddim_diffusion
        else:
            diffusion = self.diffusion

        # diffusion.p_sample_loop 用のラッパーモデル
        def model_fn(x_t, t, **kwargs):
            # x_t: [B, T, C], t: [B] or [1]
            return self.net(x_t, t, z_seq)

        # 初期ノイズ
        noise = torch.randn(shape, device=device)

        # DDPM / DDIM サンプルループ
        samples = diffusion.p_sample_loop(
            model_fn,
            shape,
            noise=noise,
            device=device,
        )  # 形状 [B, n_action_steps, action_dim]

        # そのまま「行動」として返す
        actions = samples
        return actions