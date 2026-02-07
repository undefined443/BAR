"""Pretokenization script that saves tokens in NPZ format."""
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).resolve().parent.parent))

import argparse
import datetime
import numpy as np
from PIL import Image
import torch.distributed as dist

import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.train_utils import get_pretrained_tokenizer
import utils.misc as misc
from tqdm import tqdm


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ImageFolderWithPath(datasets.ImageFolder):
    """ImageFolder that also returns the file path."""
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def get_args_parser():
    parser = argparse.ArgumentParser('Cache tokens to NPZ format', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')

    # Tokenizer parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='Image input size')
    parser.add_argument('--vae_config_path', default="", type=str,
                        help='Path to tokenizer config')
    parser.add_argument('--vae_path', default="", type=str,
                        help='Path to tokenizer checkpoint')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/imagenet', type=str,
                        help='Dataset path')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for processing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    # Caching parameters
    parser.add_argument('--cached_path', default='', help='Path to save cached tokens')

    return parser


@torch.no_grad()
def main(args):
    os.makedirs(args.cached_path, exist_ok=True)
    misc.init_distributed_mode(args)

    # Check if pretokenization is already completed
    metadata_file = os.path.join(args.cached_path, "metadata.json")
    if os.path.exists(metadata_file):
        print('=' * 80)
        print(f'Pretokenization already completed at: {args.cached_path}')
        print(f'Found existing metadata.json. Skipping pretokenization.')
        print(f'To re-run pretokenization, delete: {metadata_file}')
        print('=' * 80)
        return

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Augmentation: center crop + normalize
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    num_aug = 2  # original + flip

    dataset_train = ImageFolderWithPath(
        os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if args.vae_config_path:
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.vae_config_path)
        config.experiment.tokenizer_checkpoint = args.vae_path
        config.model.vq_model.cnn_refine = False
        # Override crop_size to match the pretokenization image size
        config.dataset.preprocessing.crop_size = args.img_size
        tokenizer = get_pretrained_tokenizer(config)
    else:
        raise NotImplementedError

    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer.to(device)

    # Get token configuration
    levels_per_channel = config.model.vq_model.get("levels_per_channel", 2)
    token_size = config.model.vq_model.token_size

    # Determine appropriate dtype for storage efficiency
    if levels_per_channel == 2:
        token_dtype = np.bool_
        print(f"Using bool dtype for binary quantization (levels_per_channel=2)")
    elif levels_per_channel <= 256:
        token_dtype = np.uint8
        print(f"Using uint8 dtype for levels_per_channel={levels_per_channel}")
    else:
        token_dtype = np.uint16
        print(f"Using uint16 dtype for levels_per_channel={levels_per_channel}")

    # Create output directory structure mirroring ImageNet
    output_train_dir = Path(args.cached_path) / "train"
    output_train_dir.mkdir(parents=True, exist_ok=True)

    # Create class subdirectories
    for class_dir in Path(args.data_path).joinpath("train").iterdir():
        if class_dir.is_dir():
            (output_train_dir / class_dir.name).mkdir(exist_ok=True)

    # Track statistics
    total_samples = 0
    total_size_bytes = 0

    # Start caching
    print(f"Start caching tokens to NPZ format, rank {args.rank}, gpu {args.gpu}")
    start_time = time.time()

    for batch_idx, (samples, targets, paths) in enumerate(tqdm(data_loader_train)):
        samples = samples.to(device, non_blocking=True)

        # Encode original + horizontal flip
        with torch.no_grad():
            _, result_dict = tokenizer.encode(samples)
            tokens_original = result_dict["min_encoding_indices"].reshape(samples.shape[0], -1)

            # Encode horizontally flipped images
            samples_flipped = torch.flip(samples, dims=[-1])
            _, result_dict_flip = tokenizer.encode(samples_flipped)
            tokens_flipped = result_dict_flip["min_encoding_indices"].reshape(samples.shape[0], -1)

        # Save each sample as a separate NPZ file
        for i in range(tokens_original.shape[0]):
            # Extract class name and filename from path
            original_path = Path(paths[i])
            class_name = original_path.parent.name
            filename = original_path.stem + ".npz"

            # Output path
            output_path = output_train_dir / class_name / filename

            # Convert to compact dtype
            tokens_orig_np = tokens_original[i].cpu().numpy().astype(token_dtype)
            tokens_flip_np = tokens_flipped[i].cpu().numpy().astype(token_dtype)

            # Save NPZ file with original and flipped tokens
            np.savez_compressed(
                output_path,
                tokens=tokens_orig_np,
                tokens_flip=tokens_flip_np,
            )

            total_samples += 1
            total_size_bytes += tokens_orig_np.nbytes + tokens_flip_np.nbytes

        # Sanity check on first batch
        if batch_idx == 0 and global_rank == 0:
            images = tokenizer.decode_tokens(tokens_original[:1])
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            images = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            Image.fromarray(images[0]).save(f"{args.cached_path}/pretokenization_debug.png")
            print(f"Saved debug image. Token shape: {tokens_orig_np.shape}, dtype: {tokens_orig_np.dtype}")
            print(f"Token min: {tokens_orig_np.min()}, max: {tokens_orig_np.max()}")

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    if misc.is_dist_avail_and_initialized():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # Print statistics
    print(f'Rank {global_rank}: Caching time {total_time_str}')
    print(f'Rank {global_rank}: Total samples: {total_samples}')
    print(f'Rank {global_rank}: Estimated size: {total_size_bytes / (1024**2):.2f} MB')
    print(f'Rank {global_rank}: Avg bytes/sample: {total_size_bytes / total_samples:.1f}')

    # Save metadata
    if global_rank == 0:
        storage_keys = ["tokens", "tokens_flip"]

        metadata = {
            "num_samples": total_samples * num_tasks,
            "num_augmentations": num_aug,
            "img_size": args.img_size,
            "format": "npz",
            "vae_config_path": args.vae_config_path,
            "vae_path": args.vae_path,
            "levels_per_channel": int(levels_per_channel),
            "token_dtype": str(token_dtype),
            "codebook_size": int(config.model.vq_model.codebook_size),
            "token_size": int(token_size),
            "storage_keys": storage_keys,
        }
        import json
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
