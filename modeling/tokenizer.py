"""CLIP text tokenizer for text diffusion models."""

import torch


class CLIPTextTokenizer:
    """Wrapper for CLIP text tokenizer for text diffusion models."""

    def __init__(self, model_name="jinaai/jina-clip-v2"):
        """Initialize CLIP text tokenizer.

        Args:
            model_name: HuggingFace model name for CLIP.
        """
        from transformers import CLIPTokenizer, CLIPModel, CLIPImageProcessor

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda")
        self.model.eval()

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    def train(self):
        """Set model to train mode (no-op for text tokenizer)."""
        return self

    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self

    def encode(self, texts, images):
        """Encode texts to binary bits and images to embeddings.

        Args:
            texts: List of text strings or single text string.
            images: List of image tensors or single image tensor.

        Returns:
            Tuple of (text_token_bits, image_embeddings).
            text_token_bits: (B, L, token_size) binary representation of token IDs
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize texts
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
                truncation=True,
            )
            token_ids = inputs["input_ids"].to(self.device)
            token_bits = self.token_ids_to_bits(token_ids)

            # Rescale images from [-1, 1] to [0, 1] for CLIP image processor
            do_rescale = True
            if isinstance(images, torch.Tensor):
                images = (images * 0.5 + 0.5).clamp(0, 1)
                do_rescale = False
            # Get embeddings from CLIP
            processed_images = self.image_processor(images, return_tensors="pt", do_rescale=do_rescale)[
                "pixel_values"
            ].to(self.device)
            outputs = self.model.vision_model(pixel_values=processed_images)
            image_embeddings = self.model.visual_projection(
                outputs.pooler_output
            )  # [batch_size, 768]

        return token_bits, image_embeddings

    def token_ids_to_bits(self, token_ids, token_size=16):
        """Convert token IDs to binary bits.

        Decomposes each integer token ID into token_size binary bits.

        Args:
            token_ids: (B, L) integer tensor with values in [0, 2^token_size - 1]
            token_size: Number of bits (default: 16)

        Returns:
            bits: (B, L, token_size) with values in {0, 1}
        """
        shifts = torch.arange(token_size, device=token_ids.device)
        # Extract each bit using right shift and bitwise AND
        bits = (token_ids.unsqueeze(-1) >> shifts) & 1
        return bits

    def bits_to_token_ids(self, bits):
        """Convert binary bits back to token IDs.

        Recomposes token_size binary bits into integer token IDs.

        Args:
            bits: (B, L, token_size) with values in {0, 1}

        Returns:
            token_ids: (B, L) integer tensor
        """
        token_size = bits.shape[-1]
        shifts = torch.arange(token_size, device=bits.device, dtype=torch.long)
        token_ids = (bits.long() << shifts).sum(dim=-1)
        return token_ids
