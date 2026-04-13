"""Training loss implementation.

Reference:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .perceptual_loss import PerceptualLoss
from .discriminator_dino import DinoDisc


# Gram Loss
class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = vgg16.features.eval()

        self.register_buffer(
            "imagenet_mean", torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "imagenet_std", torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def gram_matrix(features: torch.Tensor):
        """
        Compute Gram matrix for feature maps
        features: [N, C, H, W]
        returns: [N, C, C]
        """
        N, C, H, W = features.shape
        F_flat = features.view(N, C, H * W)  # [N, C, H*W]
        G = F_flat @ F_flat.transpose(1, 2)  # [N, C, C]
        return G / (C * H * W)  # normalize

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, L), the input concatenated image. Normalized to [-1, 1].
            target: A tensor of shape (B, C, L), the target concatenated image. Normalized to [-1, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.feature_extractor.eval()
        # [-1, 1] -> [0, 1]

        input_img = F.interpolate(
            input, size=(224, 224), mode="bilinear", align_corners=False
        )
        target_img = F.interpolate(
            target, size=(224, 224), mode="bilinear", align_corners=False
        )

        input_imgs = (input_img + 1.0) / 2.0
        target_imgs = (target_img + 1.0) / 2.0

        input_imgs = (input_imgs - self.imagenet_mean) / self.imagenet_std
        target_imgs = (target_imgs - self.imagenet_mean) / self.imagenet_std

        input_features = []
        target_features = []

        x_in = input_imgs
        x_tg = target_imgs
        for layer in self.feature_extractor:
            x_in = layer(x_in)
            x_tg = layer(x_tg)

            input_features.append(x_in)
            target_features.append(x_tg)

        total_loss = 0.0
        for f_in, f_tg in zip(input_features, target_features):
            G_in = self.gram_matrix(f_in)
            G_tg = self.gram_matrix(f_tg)
            total_loss += ((G_in - G_tg) ** 2).mean()

        return total_loss


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator.

    Reference:
        https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, config):
        """Initialize the losses module.

        Args:
            config: Configuration for the model and losses.
        """
        super().__init__()
        loss_config = config.losses
        disc_config = config.model.discriminator

        self.reconstruction_weight_l1 = loss_config.reconstruction_weight_l1
        self.reconstruction_weight_l2 = loss_config.reconstruction_weight_l2
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight

        # Discriminator: only create if discriminator_weight > 0
        self.use_discriminator = self.discriminator_weight > 0.0

        if self.use_discriminator:
            # Only use DinoDisc as discriminator
            # Download from: https://huggingface.co/nyu-visionx/RAE-collections/blob/main/discs/dino_vit_small_patch8_224.pth
            dino_ckpt_path = disc_config.get(
                "dino_ckpt_path", "assets/models/dino_vit_small_patch8_224.pth"
            )
            norm_type = disc_config.get("norm_type", "bn")
            self.discriminator = DinoDisc(
                dino_ckpt_path=dino_ckpt_path, norm_type=norm_type
            )
        else:
            print("Discriminator is disabled (discriminator_weight = 0)")
            self.discriminator = None

        # Gram loss: independent of discriminator
        self.gram_loss_weight = loss_config.get("gram_loss_weight", 0.0)
        if self.gram_loss_weight > 0:
            print(f"Gram loss enabled with weight: {self.gram_loss_weight}")
            self.gram_loss = GramLoss()
        else:
            self.gram_loss = None

        self.config = config

        # CLIP loss: frozen CLIP features are now computed in the encoder and passed via result_dict
        self.clip_loss_weight = loss_config.get("clip_loss_weight", 0.0)

        # Note: torch.compile is now done after accelerator.prepare() in train_titok.py
        # Discriminator-related methods use @torch._dynamo.disable to exclude them from compilation

    # @torch.amp.autocast(device_type='cuda', enabled=False)
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [-1, 1].
        # inputs = inputs.float()
        # reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(
                inputs, reconstructions, extra_result_dict, global_step
            )
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step: int):
        return self.use_discriminator and global_step >= self.discriminator_iter_start

    def _forward_generator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        # We always ensure the images are normalized to [-1, 1]

        reconstruction_loss = 0.0

        if self.reconstruction_weight_l1 > 0:
            reconstruction_loss += (
                F.l1_loss(inputs, reconstructions, reduction="mean")
                * self.reconstruction_weight_l1
            )
        if self.reconstruction_weight_l2 > 0:
            reconstruction_loss += (
                F.mse_loss(inputs, reconstructions, reduction="mean")
                * self.reconstruction_weight_l2
            )

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute CLIP loss using frozen CLIP features from encoder
        clip_loss = torch.zeros((), device=inputs.device)
        if self.clip_loss_weight > 0:
            clip_pred = extra_result_dict["clip_pred"]
            clip_gt = extra_result_dict["clip_gt"]
            # L2 loss between feature maps
            clip_loss = F.mse_loss(clip_pred, clip_gt, reduction="mean")

        # Compute Gram loss
        gram_loss = torch.zeros((), device=inputs.device)
        generator_loss = torch.zeros((), device=reconstructions.device)
        discriminator_factor = 0.0
        d_weight = 0.0
        if self.gram_loss_weight > 0:
            gram_loss = self.gram_loss(inputs, reconstructions)

        if self.use_discriminator:
            # Compute discriminator loss.
            generator_loss, d_weight, discriminator_factor = (
                self._compute_generator_loss(reconstructions, global_step)
            )

        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + d_weight * discriminator_factor * generator_loss
            + self.clip_loss_weight * clip_loss
            + self.gram_loss_weight * gram_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            weighted_gan_loss=(
                d_weight * discriminator_factor * generator_loss
            ).detach(),
            discriminator_factor=discriminator_factor,
            clip_loss=(self.clip_loss_weight * clip_loss).detach(),
            gram_loss=(self.gram_loss_weight * gram_loss).detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    @torch._dynamo.disable
    def _compute_generator_loss(self, reconstructions, global_step):
        """Compute generator adversarial loss."""
        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=reconstructions.device)
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        d_weight = 1.0
        d_weight *= self.discriminator_weight
        if (
            discriminator_factor > 0.0
            and self.discriminator_weight > 0.0
            and self.discriminator is not None
        ):
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)
        return generator_loss, d_weight, discriminator_factor

    @torch._dynamo.disable
    def _forward_discriminator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        # Skip discriminator training if using gram loss
        if self.discriminator is None:
            discriminator_loss = torch.zeros((), device=inputs.device)
            loss_dict = dict(
                discriminator_loss=discriminator_loss,
                logits_real=torch.zeros((), device=inputs.device),
                logits_fake=torch.zeros((), device=inputs.device),
            )
            return discriminator_loss, loss_dict

        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        # quantize reconstructions
        reconstructions = reconstructions.detach()
        reconstructions = reconstructions.clamp(-1.0, 1.0)
        reconstructions = torch.round((reconstructions + 1.0) * 127.5) / 127.5 - 1.0

        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())
        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
        )
        return discriminator_loss, loss_dict
