"""SigLIP2 vision encoder + CLIP text tokenizer for caption training."""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, CLIPTokenizer, SiglipImageProcessor

from .modules import SigLIP2Decoder


class BAR_FSQ(nn.Module):
    """Wrapper for CLIP text tokenizer + SigLIP2 vision encoder."""

    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self.decoder = SigLIP2Decoder(config)

        self.tokenizer = CLIPTokenizer.from_pretrained(config.model.vq_model.tokenizer)
        self.text_seq_len = config.model.generator.text_seq_len

        full_model = AutoModel.from_pretrained(config.model.vq_model.encoder)
        self.encoder = full_model.vision_model

        # Freeze the SigLIP2 model
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.head.requires_grad_(False)

        del full_model

        crop_size = config.dataset.preprocessing.crop_size
        self.image_processor = SiglipImageProcessor(
            size={"height": crop_size, "width": crop_size}, do_resize=False
        )

    def encode(self, texts, images):
        """Encode texts to binary bits and images to embeddings.

        Args:
            texts: List of text strings or single text string.
            images: (B, 3, H, W) tensor in [-1, 1].

        Returns:
            Tuple of (text_token_bits, image_embeddings).
            text_token_bits: (B, L, token_size) binary representation of token IDs.
            image_embeddings: (B, num_patches, hidden_size) from SigLIP2 vision model.
        """
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.text_seq_len,
                return_tensors="pt",
                truncation=True,
            )
            token_ids = inputs["input_ids"].to(images.device)
            token_bits = self._token_ids_to_bits(token_ids)

            # Rescale from [-1, 1] to [0, 1] before passing to SiglipImageProcessor
            images_rescaled = (images * 0.5 + 0.5).clamp(0, 1)
            processed_images = self.image_processor(
                images=images_rescaled, return_tensors="pt", do_rescale=False
            )["pixel_values"].to(images.device)
            outputs = self.encoder(pixel_values=processed_images)
            image_embeddings = outputs.last_hidden_state

        return token_bits, image_embeddings

    def decode_tokens(self, token_bits):
        """Decode discrete token indices to text strings.

        Args:
            token_bits: Binary representation of token IDs (B, N * token_size).

        Returns:
            texts: Decoded text strings.
        """
        # Convert bit representation back to token IDs for tokenizer decoding
        token_bits = token_bits.view(token_bits.shape[0], self.text_seq_len, -1)
        token_ids = self._bits_to_token_ids(token_bits)
        texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return texts

    def _token_ids_to_bits(self, token_ids, token_size=16):
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

    def _bits_to_token_ids(self, bits):
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

    def forward(self, input):
        return self.encode(input)

    @staticmethod
    def _character_to_image(char: str) -> Image.Image:
        """Convert a character to a PIL Image.

        Args:
            char: A single character string

        Raises:
            AssertionError: If input is not a string or not a single character
            FileNotFoundError: If Terminus font is not installed

        Returns:
            PIL Image with the character rendered
        """
        assert isinstance(char, str), "Input must be a string."
        assert len(char) == 1, "Input must be a single character."

        terminus_font = Path("/usr/share/fonts/opentype/terminus/terminus-normal.otb")
        if not terminus_font.exists():
            raise FileNotFoundError(
                "Terminus font not found. Install it with: apt install fonts-terminus"
            )

        img_size = 14
        img = Image.new("1", (img_size, img_size), color=1)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(terminus_font, size=12)
        draw.text((img_size // 2, img_size // 2), char, fill=0, font=font, anchor="mm")

        return img

    @staticmethod
    def _word_to_image(word: str) -> Image.Image:
        """Convert a word string to an image with each character rendered.

        Args:
            word: A word string

        Returns:
            PIL Image with all characters concatenated horizontally
        """
        char_images = [BAR_FSQ._character_to_image(char) for char in word]

        char_width = char_images[0].width
        char_height = char_images[0].height
        width = len(word) * char_width
        img = Image.new("1", (width, char_height), color=1)
        for i, char_img in enumerate(char_images):
            img.paste(char_img, (i * char_width, 0))

        return img

    @staticmethod
    def _tokens_to_image(tokens: list[str]) -> Image.Image:
        """Convert a list of tokens to a concatenated image.

        Args:
            tokens: List of token strings

        Returns:
            PIL Image with all tokens concatenated horizontally
        """
        word_images = [BAR_FSQ._word_to_image(token) for token in tokens]

        total_width = sum(img.width for img in word_images)
        height = word_images[0].height
        img = Image.new("1", (total_width, height), color=1)

        x_offset = 0
        for word_img in word_images:
            img.paste(word_img, (x_offset, 0))
            x_offset += word_img.width

        return img

    def texts_to_image(self, texts: list[str]) -> list[Image.Image]:
        """Convert text strings to image representations via tokenization.

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of PIL Images, one per input text
        """
        tokens_list = [
            self.tokenizer.tokenize(text)[: self.text_seq_len] for text in texts
        ]
        img_list = [BAR_FSQ._tokens_to_image(tokens) for tokens in tokens_list]

        return img_list
