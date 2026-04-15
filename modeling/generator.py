import math
import torch
import torch.nn as nn
from modeling.modules import BaseModel
from modeling.modules.blocks import init_weights, Block, RMSNorm
from modeling.modules.rope import RotaryEmbeddingFast
from .mbm_head import MaskBitModelingHead
from einops import rearrange


def build_causal_mask(seq_length):
    """Build causal attention mask for autoregressive generation."""
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


class BAR(BaseModel):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # parse the configs
        embed_dim = config.model.generator.hidden_size
        depth = config.model.generator.num_hidden_layers
        num_heads = config.model.generator.num_attention_heads
        self.num_heads = num_heads
        mlp_ratio = 4.0  # Always use 4.0 for SwiGLU

        self.original_text_seq_len = config.model.generator.text_seq_len
        self.patch_size = config.model.generator.get("patch_size", 1)

        if self.original_text_seq_len % self.patch_size != 0:
            raise ValueError(
                f"text_seq_len {self.original_text_seq_len} must be divisible by "
                f"patch_size ({self.patch_size})."
            )

        latent_width = config.model.generator.get(
            "latent_width", self.original_text_seq_len
        )

        if latent_width % self.patch_size != 0:
            raise ValueError("latent grid dimensions must be divisible by patch_size.")

        self.latent_width = latent_width
        self.text_seq_len = config.model.generator.get("text_seq_len", 77)

        target_codebook_size = config.model.generator.target_codebook_size

        generator_config = config.model.generator
        self.repeat_class_condition = generator_config.get("repeat_class_condition", 32)
        self.dropout = generator_config.get("dropout", 0.0)
        self.attn_drop = generator_config.get("attn_drop", 0.0)

        # Always use RMSNorm, SwiGLU, RoPE, and target-aware RoPE
        norm_layer = RMSNorm

        # Initialize 1D RoPE for text sequences
        head_dim = embed_dim // num_heads
        rope = RotaryEmbeddingFast(dim=head_dim)

        self.cls_token = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, 1, embed_dim)), 0.0, 0.02
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_norm=True,
                    proj_drop=self.dropout,
                    attn_drop=self.attn_drop,
                    norm_layer=norm_layer,
                    use_swiglu=True,
                    use_adaln=True,
                    rope=rope,
                    target_aware_rope=False,
                )
                for i in range(depth)
            ]
        )

        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, self.text_seq_len + 128, embed_dim)), 0.0, 0.02
        )

        self.target_aware_pos_embed = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, self.text_seq_len + 128, embed_dim)), 0.0, 0.02
        )

        self.norm = norm_layer(embed_dim)

        # FSQ uses token_size
        self.base_vq_split_channel = config.model.vq_model.token_size
        self.vq_split_channel = self.base_vq_split_channel * self.patch_size

        # MaskBitModeling head for per-token prediction
        mbm_head_config = config.model.generator.mbm_head
        self.lm_head = MaskBitModelingHead(
            num_layers=mbm_head_config.get("num_layers", 3),
            width=mbm_head_config.get("width", 2048),
            seq_len=self.vq_split_channel,
        )
        self.latent_condition_proj = nn.Linear(embed_dim, self.lm_head.width)

        self.use_checkpoint = config.model.generator.get("use_checkpoint", False)
        self.random_ratio = 0.0

        # Condition token embedding
        self.embeddings = nn.Linear(768, embed_dim)

        # Learnable unconditional/dropped condition embedding for classifier-free guidance
        self.none_condition_embedding = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # Efficient input embedding: per-channel then merge
        self.input_embeddings = nn.Embedding(
            target_codebook_size + 1, math.ceil(embed_dim / self.vq_split_channel)
        )
        self.input_merge = nn.Linear(
            math.ceil(embed_dim / self.vq_split_channel) * self.vq_split_channel,
            embed_dim,
            bias=True,
        )

        # Timestep embeddings for AdaLN conditioning
        self.timesteps_embeddings = nn.init.trunc_normal_(
            nn.Parameter(torch.zeros(1, self.text_seq_len + 100, embed_dim)), 0.0, 0.02
        )

        # Register causal attention mask as buffer
        max_seq_len = self.text_seq_len + 128
        causal_mask = build_causal_mask(max_seq_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.apply(init_weights)

        # Initialize AdaLN-Zero for transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize AdaLN-Zero for MaskBitModeling head
        for block in self.lm_head.transformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.lm_head.adaln_before_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.lm_head.adaln_before_head.adaLN_modulation[-1].bias, 0)

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()

    def sample_orders(self, x, random_ratio=None):
        """Sample orders for autoregressive generation.

        torch.compile-friendly: Always executes the same operations regardless of random_ratio value.
        This avoids recompilation when random_ratio changes during training.

        Args:
            x: Input tensor (used for batch_size and device)
            random_ratio: Probability of using random order. If None, uses self.random_ratio
        """
        if random_ratio is None:
            random_ratio = self.random_ratio

        batch_size = x.shape[0]
        device = x.device

        # Always compute both orders to make torch.compile happy
        raster_orders = (
            torch.arange(self.text_seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        random_logits = torch.rand(batch_size, self.text_seq_len, device=device)
        random_orders = torch.argsort(random_logits, dim=-1)

        # Use probabilistic blending for all cases
        # When random_ratio=0: mask is always False → returns raster_orders
        # When random_ratio=1: mask is always True → returns random_orders
        # When 0 < random_ratio < 1: blends based on per-batch random mask
        random_mask = (torch.rand(batch_size, 1, device=device) < random_ratio).expand(
            -1, self.text_seq_len
        )
        return torch.where(random_mask, random_orders, raster_orders)

    def set_random_ratio(self, new_ratio):
        self.random_ratio = new_ratio

    def get_raster_orders(self, x):
        batch_size = x.shape[0]
        # torch.compile-friendly: use expand instead of list comprehension
        raster_orders = (
            torch.arange(self.text_seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        return raster_orders

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        )
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = (
            torch.arange(batch_size, device=shuffled_x.device)
            .unsqueeze(1)
            .expand(-1, seq_len)
        )
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def _downsample_tokens(self, tokens):
        if self.patch_size == 1:
            return tokens
        sw = self.patch_size
        w = self.latent_width // sw
        return rearrange(
            tokens,
            "b (w sw) n -> b w (sw n)",
            w=w,
            sw=sw,
        )

    def _upsample_tokens(self, tokens):
        if self.patch_size == 1:
            return tokens
        sw = self.patch_size
        w = self.latent_width // sw
        return rearrange(
            tokens,
            "b w (sw n) -> b (w sw) n",
            w=w,
            sw=sw,
        )

    def preprocess_condition(self, condition, cond_drop_prob=0.0):
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        condition = torch.where(
            drop_label_mask, self.none_condition_embedding, condition
        )
        return condition

    def get_none_condition(self, condition):
        return self.none_condition_embedding.expand(condition.shape[0], -1)

    def forward(self, input_ids, condition, random_ratio=None, orders=None):
        """Forward pass through BAR model.

        Args:
            input_ids: Input token IDs
            condition: Condition embeddings
            random_ratio: Ratio for random vs raster ordering. If None, uses self.random_ratio.
                         DEPRECATED: Use orders parameter instead for better performance.
            orders: Pre-sampled orders. If provided, used directly (recommended for training).
                   If None, will sample orders based on random_ratio.
        """
        # If orders are pre-sampled (recommended), use them directly
        if orders is None:
            # Fall back to sampling orders based on random_ratio
            if random_ratio is None:
                random_ratio = self.random_ratio
            orders = self.sample_orders(input_ids, random_ratio=random_ratio)

        return self.forward_fn(input_ids, condition, orders)

    def embed_input_ids(self, input_ids):
        """Embed input_ids using efficient per-channel embedding.

        Args:
            input_ids: [B, L, vq_split_channel]
        Returns:
            input_embed: [B, L, embed_dim]
        """
        input_embed = self.input_embeddings(input_ids)
        input_embed = rearrange(input_embed, "b l n c -> b l (n c)")
        input_embed = self.input_merge(input_embed)
        return input_embed

    def forward_fn(
        self,
        input_ids,
        condition,
        orders=None,
        is_sampling=False,
        apply_generator_downsample=None,
    ):
        if apply_generator_downsample is None:
            apply_generator_downsample = not is_sampling
        if apply_generator_downsample and self.patch_size > 1:
            input_ids = input_ids.reshape(
                input_ids.shape[0],
                self.original_text_seq_len,
                self.base_vq_split_channel,
            )
            input_ids = self._downsample_tokens(input_ids)
        else:
            input_ids = input_ids.reshape(input_ids.shape[0], -1, self.vq_split_channel)

        if orders is None:
            orders = self.get_raster_orders(input_ids)

        labels = input_ids.clone()  # Clean tokens for loss computation

        condition_token = self.embeddings(condition)
        input_embed = self.embed_input_ids(input_ids)

        # Repeat condition_token according to repeat_class_condition
        # Shape: (B, repeat_class_condition, embed_dim)
        repeated_condition_token = condition_token.unsqueeze(1).repeat(
            1, self.repeat_class_condition, 1
        )
        embeddings = torch.cat([repeated_condition_token, input_embed], dim=1)

        # Prepare positional embeddings with shuffling
        pos_embed = self.pos_embed.repeat(input_ids.shape[0], 1, 1)
        prefix = (
            1 + self.repeat_class_condition
        )  # cls_token + repeated condition tokens
        pos_embed_prefix = pos_embed[:, :prefix]
        pos_embed_postfix = self.shuffle(
            pos_embed[:, prefix : prefix + self.text_seq_len], orders
        )

        # Prepare target-aware positional embeddings
        # Each position should have the embedding for the spatial position it will predict
        target_aware_pos_embed = self.target_aware_pos_embed.repeat(
            input_ids.shape[0], 1, 1
        )

        # Create target order: [orders[0], orders[1], orders[2], ..., orders[-1], orders[-1]]
        # This aligns with target_rope_order logic
        batch_size = input_ids.shape[0]
        target_orders = torch.cat(
            [orders[:, 1:], orders[:, -1:]], dim=1
        )  # Shift by 1: [orders[1], ..., orders[-1], orders[-1]]

        # Shuffle target_aware_pos_embed using target_orders to get embeddings for prediction targets
        target_aware_pos_embed_shuffled = self.shuffle(
            target_aware_pos_embed[:, prefix : prefix + self.text_seq_len],
            target_orders,
        )

        # For prefix positions, use embedding for orders[0] (first token to predict)
        # Extract embedding for orders[0] for each batch element
        first_target_embed = (
            target_aware_pos_embed[
                torch.arange(batch_size, device=orders.device), orders[:, 0] + prefix
            ]
            .unsqueeze(1)
            .expand(-1, prefix, -1)
        )  # [B, prefix, dim]

        # Concatenate: [emb[orders[0]] repeated prefix times, emb[orders[1]], emb[orders[2]], ..., emb[orders[-1]]]
        target_aware_pos_embed_full = torch.cat(
            [first_target_embed, target_aware_pos_embed_shuffled], dim=1
        )

        if not is_sampling:
            labels = self.shuffle(labels, orders)
            # Keep repeated condition tokens (first repeat_class_condition positions) together, shuffle the rest
            embeddings = torch.cat(
                [
                    embeddings[:, : self.repeat_class_condition],
                    self.shuffle(embeddings[:, self.repeat_class_condition :], orders),
                ],
                dim=1,
            )

        cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
        x = torch.cat((cls_tokens, embeddings), dim=1)

        x = x + torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)[:, : x.shape[1]]

        # Add target-aware positional embeddings
        x = x + target_aware_pos_embed_full[:, : x.shape[1]]

        time_embeddings = self.timesteps_embeddings[:, : x.shape[1]]
        # Expand condition_token for each repeated position and add time embeddings
        # Shape: (B, 1, embed_dim) -> (B, repeat_class_condition, embed_dim)
        condition_token_with_time = condition_token.unsqueeze(1) + time_embeddings
        # Expand to full sequence for use in blocks
        # condition_token_for_blocks = F.silu(condition_token_for_blocks)

        # Prepare causal attention mask for current sequence length
        seq_len = x.shape[1]
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Prepare RoPE order for shuffled tokens (always use RoPE with target-aware positions)
        # Sequence structure: [cls_token (pos 0), condition_tokens (pos 1..repeat_class_condition), shuffled_tokens (pos prefix+)]
        if orders is not None:
            # Create position indices for the full sequence
            # Prefix tokens keep their positions [0, 1, ..., prefix-1], shuffled tokens get orders + prefix
            batch_size = x.shape[0]
            rope_order_full = torch.cat(
                [
                    torch.arange(prefix, device=orders.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1),  # [B, prefix]
                    orders
                    + prefix,  # [B, text_seq_len] with positions [prefix, prefix+1, ...]
                ],
                dim=1,
            )  # [B, seq_len]
            rope_order = rope_order_full[
                :, :seq_len
            ]  # Slice to current sequence length

            # Prepare target-aware RoPE order (spatial position of token being predicted)
            # After autoregressive slicing (x[:, prefix-1:] then x[:, :-1]):
            # Position 0 predicts token at spatial position orders[0]
            # Position i predicts token at spatial position orders[i]
            # So target positions are: [orders[0], orders[1], ..., orders[N-1]]
            # We need to construct this for the full sequence before slicing

            # For all prefix positions (0..prefix-1), target is orders[0]
            # For position prefix+i, target is orders[i+1] (or orders[i] for the last one, doesn't matter)
            prefix_targets = (orders[:, 0:1] + prefix).expand(
                -1, prefix
            )  # [B, prefix] all point to orders[0]
            # Shift orders by 1: [orders[1], orders[2], ..., orders[-1], orders[-1]]
            shuffled_targets = (
                torch.cat([orders[:, 1:], orders[:, -1:]], dim=1) + prefix
            )  # [B, text_seq_len]

            target_rope_order_full = torch.cat(
                [prefix_targets, shuffled_targets], dim=1
            )  # [B, prefix + text_seq_len]
            target_rope_order = target_rope_order_full[
                :, :seq_len
            ]  # Slice to current sequence length
        else:
            rope_order = None
            target_rope_order = None

        if self.blocks[0].attn.kv_cache and self.blocks[0].attn.k_cache is not None:
            x = x[:, -1:]
            condition_token_with_time = condition_token_with_time[:, -1:]
            attn_mask = None
            # Keep rope_order for the last position to support random order generation with KV cache
            if rope_order is not None:
                rope_order = rope_order[:, -1:]
            if target_rope_order is not None:
                target_rope_order = target_rope_order[:, -1:]

        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    blk.forward,
                    x,
                    attn_mask,
                    condition_token_with_time,
                    rope_order,
                    target_rope_order,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    attn_mask=attn_mask,
                    c=condition_token_with_time,
                    rope_order=rope_order,
                    target_rope_order=target_rope_order,
                )

        if not self.blocks[0].attn.kv_cache:
            x = x[:, prefix - 1 :]
            condition_token_with_time = condition_token_with_time[:, prefix - 1 :]

        x = self.norm(x)
        x = self.latent_condition_proj(x)

        if is_sampling:
            return x
        else:
            x = x[:, :-1]
            target = labels.reshape(-1, self.vq_split_channel)
            conditions = x.reshape(-1, 1, self.lm_head.width)

            loss = self.lm_head(target=target, conditions=conditions)
            return loss

    @torch.no_grad()
    def generate(
        self,
        condition,
        guidance_scale,
        randomize_temperature,
        tokens_allocation=[4, 4, 4, 4],
        kv_cache=True,
        sample_with_random_order=False,
        **kwargs,
    ):

        condition = self.preprocess_condition(condition, cond_drop_prob=0.0)
        device = condition.device
        num_samples = condition.shape[0]
        ids = torch.full((num_samples, 0), -1, device=device)

        if kv_cache:
            self.enable_kv_cache()

        # Always generate orders (raster or random)
        if sample_with_random_order:
            orders = self.sample_orders(ids)
        else:
            # Raster order: sequential positions [0, 1, 2, ...]
            orders = self.get_raster_orders(ids)

        # Determine where to apply CFG
        use_cfg = guidance_scale > 1.0

        if use_cfg:
            cfg_orders = torch.cat([orders, orders], dim=0)
        else:
            cfg_orders = orders

        for step in range(self.text_seq_len):
            ratio = step / self.text_seq_len

            # Linear CFG annealing: cfg_scale increases from 1.0 to guidance_scale
            if guidance_scale <= 1.0:
                cfg_scale = 1.0
            else:
                cfg_scale = 1.0 + (guidance_scale - 1.0) * ratio

            # Forward pass with or without CFG at BAR level
            if use_cfg:
                latent_conditions = self.forward_fn(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, self.get_none_condition(condition)], dim=0),
                    orders=cfg_orders,
                    is_sampling=True,
                )[:, -1].reshape(-1, 1, self.lm_head.width)
            else:
                latent_conditions = self.forward_fn(
                    ids, condition, orders=cfg_orders, is_sampling=True
                )[:, -1].reshape(-1, 1, self.lm_head.width)

            sampled = self.lm_head.sample(
                conditions=latent_conditions,
                guidance_scale=cfg_scale,
                randomize_temperature=randomize_temperature,
                tokens_allocation=tokens_allocation,
                use_cfg=use_cfg,
            )
            ids = torch.cat((ids, sampled), dim=-1)

        self.disable_kv_cache()

        ids = ids.view(ids.shape[0], self.text_seq_len, self.vq_split_channel)

        if orders is not None:
            # at last, unshuffle the ids
            ids = self.unshuffle(ids, orders)

        if self.patch_size > 1:
            ids = self._upsample_tokens(ids)

        return ids.view(ids.shape[0], -1)
