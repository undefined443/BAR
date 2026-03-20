import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.modules.blocks import modulate, FinalLayer, SwiGLUFFN, RMSNorm
import math


# This block is used to build adaMLP
class ResBlock(nn.Module):
    """Residual block with AdaLN modulation and SwiGLU activation."""

    def __init__(self, channels, norm_layer=RMSNorm):
        super().__init__()
        self.channels = channels
        self.in_ln = norm_layer(channels)
        # Always use SwiGLU with mlp_ratio=4.0
        self.mlp = SwiGLUFFN(in_features=channels, hidden_features=int(2/3 * int(channels * 4.0)))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, attn_mask=None, c=None):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier Embedding for timesteps."""

    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W
        t = t[:, None] * W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class MaskBitModelingHead(nn.Module):
    """Mask Bit Modeling head for token prediction.

    Simplified version with hardcoded best-performing configurations:
    - Always uses SwiGLU activation and RMSNorm
    - Logit-normal timestep sampling with mean=0.0, std=0.8
    - Identity timestep-to-mask-ratio mapping
    - Binary quantization (target_codebook_size=2)
    """

    def __init__(self, target_codebook_size=2, num_layers=3, width=2048, seq_len=16, token_vocab_size=None):
        super(MaskBitModelingHead, self).__init__()

        self.num_layers = num_layers
        self.width = width
        self.seq_len = seq_len
        self.token_vocab_size = token_vocab_size or (2 ** seq_len)

        # Always use RMSNorm and SwiGLU
        norm_layer = RMSNorm

        # Input projection from noisy categorical token states [B, vocab_size] -> [B, width]
        self.input_proj = nn.Linear(self.token_vocab_size, self.width, bias=True)
        self.ln_pre = norm_layer(self.width)

        # Transformer blocks with SwiGLU (mlp_ratio=4.0 hardcoded)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResBlock(
                channels=self.width,
                norm_layer=norm_layer,
            ))

        # Output projection: predict token logits [B, width] -> [B, vocab_size]
        self.output_embed = nn.Linear(self.width, self.token_vocab_size, bias=True)

        self.apply(self._init_weights)

        self.t_embedder = GaussianFourierEmbedding(self.width)
        self.adaln_before_head = FinalLayer(self.width, norm_layer)
        basis = 2 ** torch.arange(self.seq_len, dtype=torch.long)
        self.register_buffer("bit_basis", basis, persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            m.weight.data = nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _sample_timesteps(self, batch_size, device):
        """Sample timesteps using logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to place the timesteps on

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Sample from N(0.0, 0.8²) and apply sigmoid
        z = torch.randn((batch_size,), device=device) * 0.8 + 0.0
        return torch.sigmoid(z)

    def _get_alpha_bar(self, t):
        """Cosine noise schedule: alpha_bar_t = cos^2(pi * t / 2).

        Args:
            t: Tensor of timesteps in [0, 1], shape [B]

        Returns:
            alpha_bar: Tensor of shape [B], values in [0, 1]
        """
        return torch.cos(t * math.pi / 2) ** 2

    def _add_noise(self, x0, t):
        """Forward diffusion process in continuous token-state space."""
        alpha_bar = self._get_alpha_bar(t)[:, None]  # [B, 1]
        eps = torch.randn_like(x0)
        x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
        return x_t, eps, alpha_bar

    def bits_to_token_ids(self, target):
        """Convert per-token binary codes to integer token IDs."""
        target = target.long()
        return (target * self.bit_basis).sum(dim=-1)

    def token_ids_to_bits(self, token_ids):
        """Convert integer token IDs to per-token binary codes."""
        token_ids = token_ids.long().unsqueeze(-1)
        bit_positions = torch.arange(self.seq_len, device=token_ids.device)
        return ((token_ids >> bit_positions) & 1).long()

    def forward_fn(self, x_t, conditions, timesteps):
        """Run MBM head network.

        Args:
            x_t: Noisy categorical token states [B, vocab_size] in continuous space
            conditions: Latent conditions from BAR [B, 1, width]
            timesteps: Diffusion timesteps [B] in [0, 1]

        Returns:
            token_logits: Predicted token logits [B, vocab_size]
        """
        inputs = self.input_proj(x_t)  # [B, width]

        t_emb = self.t_embedder(timesteps).unsqueeze(1)
        y_emb = conditions
        s = F.silu(t_emb + y_emb).squeeze(1)

        x = self.ln_pre(inputs)
        for i in range(self.num_layers):
            x = self.transformer[i](x, c=s)

        x = self.adaln_before_head(x, s)
        return self.output_embed(x)  # [B, vocab_size]

    def forward(self, target, conditions):
        """Forward pass through MBM head.

        Args:
            target: Target tokens [B, seq_len] in {0, 1}
            conditions: Latent conditions from BAR [B, 1, width]
        """
        token_ids = self.bits_to_token_ids(target)
        x0 = F.one_hot(token_ids, num_classes=self.token_vocab_size).to(dtype=conditions.dtype)
        batch_size, device = x0.shape[0], x0.device

        timesteps = self._sample_timesteps(batch_size, device)
        x_t, _, _ = self._add_noise(x0, timesteps)

        out = self.forward_fn(x_t, conditions, timesteps)

        with torch.amp.autocast('cuda', enabled=False):
            loss = F.cross_entropy(out.float(), token_ids)
        return loss

    @torch.no_grad()
    def sample(self, conditions, guidance_scale=3.0, num_steps=50, use_cfg=False):
        """Sample tokens using DDIM denoising.

        Args:
            conditions: Latent conditions. When CFG is used (use_cfg=True),
                       conditions are doubled: [batch_size*2, 1, width] where first half
                       is conditional, second half is unconditional.
            guidance_scale: CFG guidance scale. Used when use_cfg=True.
            num_steps: Number of DDIM denoising steps.
            use_cfg: Whether to apply CFG. If True, conditions should be doubled.
        """
        device = conditions.device

        if use_cfg:
            batch_size = conditions.shape[0] // 2
        else:
            batch_size = conditions.shape[0]

        # Start from pure Gaussian noise in continuous token-state space.
        x_t = torch.randn(batch_size, self.token_vocab_size, device=device, dtype=conditions.dtype)

        # t from 1.0 (noisy) to 0.0 (clean), num_steps+1 points
        step_ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=conditions.dtype)

        for i in range(num_steps):
            t_cur = step_ts[i].view(1).expand(batch_size)    # [B]
            t_next = step_ts[i + 1].view(1).expand(batch_size)  # [B]

            alpha_bar_cur = self._get_alpha_bar(t_cur)[:, None]   # [B, 1]
            alpha_bar_next = self._get_alpha_bar(t_next)[:, None]  # [B, 1]

            if use_cfg:
                t_cat = t_cur.repeat(2)
                x_t_cat = x_t.repeat(2, 1)
                out_both = self.forward_fn(x_t_cat, conditions, t_cat)
                out_cond, out_uncond = out_both.chunk(2, dim=0)
                out = out_uncond + guidance_scale * (out_cond - out_uncond)
            else:
                out = self.forward_fn(x_t, conditions, t_cur)

            token_ids = out.argmax(dim=-1)
            x0_pred = F.one_hot(token_ids, num_classes=self.token_vocab_size).to(dtype=x_t.dtype)

            eps_pred = (x_t - alpha_bar_cur.sqrt() * x0_pred) / (1 - alpha_bar_cur).sqrt().clamp(min=1e-8)
            x_t = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * eps_pred

        token_ids = x_t.argmax(dim=-1)
        return self.token_ids_to_bits(token_ids)

