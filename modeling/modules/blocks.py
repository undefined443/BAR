import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from timm.layers import Mlp
from einops import rearrange
from einops.layers.torch import Rearrange
from transformers import AutoModel


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            output = output * self.weight
        return output.to(input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features = None,
        out_features = None,
        drop = 0.0,
        bias = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.drop(hidden)  # Apply dropout after hidden and before w3
        return self.w3(hidden)

# weight init
def init_weights(module):
    if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or
     isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)

# attention layer with KV cache supported
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer = nn.LayerNorm,
            rope: nn.Module = None,
            target_aware_rope: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None
        self.rope = rope
        self.target_aware_rope = target_aware_rope
        if target_aware_rope:
            assert num_heads % 2 == 0, 'num_heads must be even for target_aware_rope'

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, attn_mask=None, rope_order=None, target_rope_order=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE if provided
        # rope_order contains the position indices for each token
        if self.rope is not None:
            # q and k shape: (B, num_heads, N, head_dim)
            if self.target_aware_rope and target_rope_order is not None:
                # Split Q heads: first half uses current_location, second half uses target_location
                half_heads = self.num_heads // 2
                q_current = q[:, :half_heads]  # First half of heads
                q_target = q[:, half_heads:]   # Second half of heads

                # Apply RoPE with current location to first half
                q_current = self.rope(q_current, rope_order=rope_order)
                # Apply RoPE with target location to second half
                q_target = self.rope(q_target, rope_order=target_rope_order)

                # Concatenate back
                q = torch.cat([q_current, q_target], dim=1)

                # K always uses current location
                k = self.rope(k, rope_order=rope_order)
            else:
                # Standard RoPE: apply same rope_order to both q and k
                q = self.rope(q, rope_order=rope_order)
                k = self.rope(k, rope_order=rope_order)

        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                assert N in [1, 2], f"x.shape {x.shape}"
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache

            k = k_cache
            v = v_cache

        # Use PyTorch SDPA (scaled_dot_product_attention)
        # Let PyTorch choose the best backend automatically
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    def __init__(self, dim, norm_layer=RMSNorm):
        super().__init__()
        self.norm_final = norm_layer(dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
    

# basic transformer block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            use_swiglu: bool = False,
            use_adaln: bool = False,
            norm_layer = RMSNorm,
            rope: nn.Module = None,
            target_aware_rope: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rope=rope,
            target_aware_rope=target_aware_rope,
        )

        self.norm2 = norm_layer(dim)

        if use_swiglu:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=int(2/3 * int(dim * mlp_ratio)),
                drop=proj_drop,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )

        self.use_adaln = use_adaln
        if self.use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, 6 * dim, bias=True)
            )


    def forward(self, x: torch.Tensor, attn_mask=None, c = None, rope_order=None, target_rope_order=None) -> torch.Tensor:
        if self.use_adaln:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask, rope_order=rope_order, target_rope_order=target_rope_order)
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x), attn_mask=attn_mask, rope_order=rope_order, target_rope_order=target_rope_order)
            x = x + self.mlp(self.norm2(x))
        return x


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class SigLIP2Encoder(nn.Module):
    """
    Vision encoder using SigLIP2 architecture with dual encoder design.
    - Uses frozen SigLIP2 embeddings and a frozen transformer stack
    - Uses a separate trainable transformer stack
    - Returns features from both encoders
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_name = "google/siglip2-so400m-patch16-naflex"

        # Load pretrained SigLIP2 vision model (frozen)
        full_model = AutoModel.from_pretrained(model_name)
        self.siglip = full_model.vision_model

        # Freeze the SigLIP2 model
        self.siglip.eval()
        self.siglip.requires_grad_(False)
        # Head is never used, disable gradients
        self.siglip.head.requires_grad_(False)

        del full_model

        # Get architecture parameters
        self.width = self.siglip.config.hidden_size
        self.patch_size = 16
        self.original_image_size = config.dataset.preprocessing.crop_size
        self.image_size = self.original_image_size
        self.grid_size = self.image_size // self.patch_size

        # Use the pretrained SigLIP2 encoder as frozen encoder
        self.frozen_encoder = self.siglip.encoder
        self.frozen_encoder.eval()
        self.frozen_encoder.requires_grad_(False)

        # Deep copy the pretrained SigLIP2 encoder for trainable encoder
        self.trainable_encoder = copy.deepcopy(self.siglip.encoder)
        self.trainable_encoder.train()
        self.trainable_encoder.requires_grad_(True)

        # Post layer norms - frozen uses the pretrained one
        self.frozen_post_layernorm = self.siglip.post_layernorm
        self.frozen_post_layernorm.eval()
        self.frozen_post_layernorm.requires_grad_(False)

        # Trainable post layernorm - deep copy from pretrained
        self.trainable_post_layernorm = copy.deepcopy(self.siglip.post_layernorm)
        self.trainable_post_layernorm.train()
        self.trainable_post_layernorm.requires_grad_(True)

        # Register spatial_shapes as a buffer for torch.compile compatibility
        # Shape: (1, 2) containing [grid_size, grid_size]
        spatial_shapes_template = torch.tensor([[self.grid_size, self.grid_size]], dtype=torch.long)
        self.register_buffer('spatial_shapes_template', spatial_shapes_template, persistent=False)

    def _compute_frozen_features(self, hidden_states):
        """
        Helper function to compute frozen CLIP features.

        Args:
            hidden_states: Embeddings from SigLIP2 embeddings layer (B, num_patches, C)

        Returns:
            frozen_features: Features from frozen encoder (B, num_patches, C)
        """
        frozen_hidden = hidden_states
        for frozen_layer in self.frozen_encoder.layers:
            frozen_hidden = frozen_layer(frozen_hidden, attention_mask=None)
            if isinstance(frozen_hidden, tuple):
                frozen_hidden = frozen_hidden[0]

        frozen_features = self.frozen_post_layernorm(frozen_hidden)
        return frozen_features

    def forward(self, pixel_values, return_clip_gt=True):
        """
        Args:
            pixel_values: Input images, shape (B, 3, H, W), normalized to [-1, 1]
            return_clip_gt: If True, also compute and return frozen CLIP features

        Returns:
            If return_clip_gt=True:
                Tuple of (trainable_features, frozen_features):
                - trainable_features: Features from trainable encoder (B, num_patches, C)
                - frozen_features: Features from frozen encoder (B, num_patches, C)
            If return_clip_gt=False:
                trainable_features: Features from trainable encoder (B, num_patches, C)
        """
        # Ensure frozen components stay frozen
        self.siglip.eval()
        self.siglip.requires_grad_(False)

        # Prepare spatial shapes for embeddings
        # Use expand instead of repeat for torch.compile compatibility
        batch_size = pixel_values.shape[0]
        spatial_shapes = self.spatial_shapes_template.expand(batch_size, -1)

        # Flatten for naflex architecture
        pixel_values = rearrange(pixel_values, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        hidden_states = self.siglip.embeddings(pixel_values=pixel_values, spatial_shapes=spatial_shapes)

        # Process through trainable encoder
        trainable_hidden = hidden_states
        for trainable_layer in self.trainable_encoder.layers:
            trainable_hidden = trainable_layer(trainable_hidden, attention_mask=None)
            if isinstance(trainable_hidden, tuple):
                trainable_hidden = trainable_hidden[0]

        trainable_features = self.trainable_post_layernorm(trainable_hidden)

        # Optionally process through frozen encoder for ground truth
        if return_clip_gt:
            frozen_features = self._compute_frozen_features(hidden_states)
            return trainable_features, frozen_features
        else:
            return trainable_features


class SigLIP2Decoder(nn.Module):
    """
    Decoder that works with SigLIP2Encoder outputs.
    Reconstructs from 2D tokens (grid_size^2 image tokens).

    Features:
    - Learnable positional embeddings
    - LayerNorm and standard MLP
    - CLIP alignment for distillation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size

        # ViT-L architecture parameters
        self.width = 1024
        self.num_layers = 24
        self.num_heads = 16
        self.mlp_ratio = 4.0

        # Always use LayerNorm and standard MLP
        norm_layer = nn.LayerNorm

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size ** 2 + 1, self.width))

        self.ln_pre = norm_layer(self.width)
        self.transformer = nn.ModuleList()

        for i in range(self.num_layers):
            self.transformer.append(Block(
                dim=self.width,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=0.,
                attn_drop=0.,
                norm_layer=norm_layer,
                use_swiglu=False,
                use_adaln=False,  # No AdaLN for decoder
            ))
        self.ln_post = norm_layer(self.width)

        # CLIP target width (always 1152 for so400m)
        target_width = 1152

        # RGB reconstruction
        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                p1=self.patch_size, p2=self.patch_size),
        )
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

        # CLIP alignment
        loss_config = config.get("losses", {"clip_loss_weight": 0.0})
        self.clip_align = loss_config.get("clip_loss_weight", 0.0) > 0
        self.clip_align_layer_id = config.model.vq_model.get("clip_align_layer_id", 2)
        self.clip_projector = nn.Sequential(
            nn.Linear(self.width, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, target_width),
        ) if self.clip_align else None

        # Potentially use checkpointing
        self.use_checkpoint = config.model.vq_model.get("use_checkpoint", False)
        self.attn_mask = None

    def forward(self, z_quantized):
        """
        Args:
            z_quantized: 2D tokens (B, grid_size^2, C)

        Returns:
            (reconstructed_image, clip_pred) where image is (B, 3, H, W)
        """
        clip_pred = None
        x = z_quantized
        batchsize, seq_len, _ = x.shape

        # Add class embedding and positional encoding
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        attn_mask = self.attn_mask
        x = self.ln_pre(x)

        # Forward through transformer layers
        for i in range(self.num_layers):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.transformer[i].forward, x, attn_mask, use_reentrant=False)
            else:
                x = self.transformer[i](x, attn_mask=attn_mask)

            if self.clip_align and i == self.clip_align_layer_id:
                clip_pred = self.clip_projector(x)[:, 1:1+self.grid_size**2]

        # Extract image tokens (remove cls embedding)
        x = x[:, 1:1+self.grid_size**2]
        x = self.ln_post(x)

        # Reconstruct RGB image
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x, clip_pred
