# Autoregressive Image Generation with Masked Bit Modeling

<div align="center">

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://bar-gen.github.io/)&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2602.09024)&nbsp;&nbsp;

</div>

## Introduction

Visual generative models have driven remarkable progress across computer vision tasks. A central component of these systems is **visual tokenization**, which compresses high-dimensional pixel inputs into compact latent representations. Depending on quantization strategies, visual tokenization pipelines can be categorized into *discrete* and *continuous* approaches. While continuous tokenizers with diffusion models currently dominate visual generation, we investigate whether this gap is fundamental or merely a consequence of design choices.

We introduce **BAR** (*masked **B**it **A**uto**R**egressive modeling*), a strong discrete visual generation framework that challenges the prevailing dominance of continuous pipelines. Our key insight is that the commonly observed inferior performance of discrete tokenizers is largely attributable to their substantially higher compression ratios, which lead to severe information loss.

### Key Findings

**1. Discrete Tokenizers Beat Continuous Tokenizers**
By scaling the codebook size to match the bit budget of continuous tokenizers, we demonstrate that discrete tokenizers can achieve competitive or superior reconstruction quality.

**2. Discrete Autoregressive Models Beat Diffusion**
We address the vocabulary scaling problem by replacing the standard linear prediction head with a lightweight **Masked Bit Modeling (MBM)** head. This design enables training with arbitrary codebook sizes while maintaining superior generation quality. BAR achieves state-of-the-art gFID of **0.99** on ImageNet-256, surpassing leading continuous and discrete approaches.

**3. Superior Efficiency**
BAR offers reduced sampling costs and faster convergence compared to diffusion models. Our efficient variant BAR-B/4 achieves 2.94× speedup over MeanFlow while matching its performance, and BAR-B is 3.68× faster than RAE while achieving same FID (1.13 vs 1.13).

<p align="center">
  <img src="assets/teaser1.png" alt="Unified Bit Budget Metric" width=90%>
</p>
<p align="center">
<em>By measuring information capacity in bits, we enable direct comparison between discrete and continuous tokenizers. Our discrete tokenizer (BAR-FSQ) scales smoothly with codebook size and matches or surpasses continuous tokenizers at sufficient bit budgets.</em>
</p>

<p align="center">
  <img src="assets/teaser2.png" alt="Performance and Efficiency" width=90%>
</p>
<p align="center">
<em>BAR establishes a new state of the art across both discrete and continuous paradigms. With only 415M parameters, BAR-B achieves gFID of 1.13, matching RAE while being significantly faster. BAR-L (1.1B parameters) achieves a record gFID of 0.99.</em>
</p>

## Method Overview

### BAR Framework

BAR decomposes autoregressive visual generation into two stages:

1. **Context Modeling**: An autoregressive transformer captures global structure via causal attention, generating latent conditions for each token position.

2. **Token Prediction**: Instead of a standard linear head that scales poorly with large vocabularies, we introduce a **Masked Bit Modeling (MBM)** head that generates discrete tokens through progressive bit-wise unmasking.

### Masked Bit Modeling Head

The MBM head treats token prediction as a conditional generation task rather than massive classification:
- **Scalability**: Memory complexity reduces from O(C) to O(log₂ C), where C is codebook size
- **Quality**: Bit-wise masking acts as a strong regularizer, consistently improving generation quality
- **Flexibility**: Supports arbitrary codebook sizes without architectural changes

At inference, the MBM head generates each token via iterative unmasking with a configurable schedule (e.g., [2, 2, 5, 7] unmasks 2, 2, 5, then 7 bits across 4 steps).

## Model Zoo

We provide pretrained generator models:

| Model | Config | Size | gFID | IS |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| BAR-B/4 | [bar_b_patch4.yaml](configs/generator/bar_b_patch4.yaml) | 416M | 2.34 | 274.7 |
| BAR-B/2 | [bar_b_patch2.yaml](configs/generator/bar_b_patch2.yaml) | 415M | 1.35 | 293.4 |
| BAR-B | [bar_b.yaml](configs/generator/bar_b.yaml) | 415M | 1.13 | 289.0 |
| BAR-L | [bar_l.yaml](configs/generator/bar_l.yaml) | 1.1B | 0.99 | 296.9 |
| BAR-L-res512 | [bar_l_res512.yaml](configs/generator/bar_l_res512.yaml) | 1.1B | 1.09 | 311.1 |


## Preparation

### Environment Setup
```shell
uv sync && source .venv/bin/activate
```

### Checkpoint Setup

**Download Model Checkpoints from HuggingFace:**
```bash
hf download FAR-Amazon/BAR-collections \
  --local-dir assets
```

This will download all pretrained models (generators, tokenizers, and discriminator weights) to the `assets/` folder.

## Quick Start

### Training

**Data Preparation:**

We use webdataset format for data loading. To begin with, it is needed to convert the dataset into webdataset format. An example script to convert ImageNet to wds format is provided [here](https://github.com/bytedance/1d-tokenizer/blob/main/data/convert_imagenet_to_wds.py).

**Train BAR-FSQ Tokenizer:**
```bash
WANDB_MODE=offline WORKSPACE=/path/to/workspace \
accelerate launch --num_machines=1 --num_processes=8 \
    --machine_rank=0 --main_process_ip=127.0.0.1 \
    --main_process_port=9999 --same_network \
    scripts/train_bar_fsq.py \
    config=configs/tokenizer/bar_fsq_16bits.yaml \
    'dataset.params.train_shards_path_or_url=/path/to/imagenet-train-{000000..000320}.tar' \
    'dataset.params.eval_shards_path_or_url=/path/to/imagenet-val-{000000..000049}.tar'
```

**Train BAR Generator:**

Optionally pretokenize ImageNet dataset to NPZ format for faster training:
```bash
# Note: data_path should point to ImageNet in original jpeg format (train/ and val/ folders)
torchrun --nproc-per-node=8 scripts/pretokenization.py \
    --img_size 256 \
    --batch_size 32 \
    --vae_config_path configs/tokenizer/bar_fsq_16bits.yaml \
    --vae_path assets/tokenizer/bar_fsq_16bits.bin \
    --data_path /path/to/imagenet \
    --cached_path ./pretokenized_npz
```

Train the generator:

With pretokenization:
```bash
WANDB_MODE=offline WORKSPACE=/path/to/workspace \
accelerate launch --num_machines=N --num_processes=$((8*N)) \
    --machine_rank=RANK --main_process_ip=MAIN_IP \
    --main_process_port=9999 --same_network \
    scripts/train_bar.py \
    config=configs/generator/bar_b_patch4.yaml \
    dataset.params.pretokenization=./pretokenized_npz \
    training.per_gpu_batch_size=$((2048 / (8*N)))
```

Without pretokenization:
```bash
WANDB_MODE=offline WORKSPACE=/path/to/workspace \
accelerate launch --num_machines=N --num_processes=$((8*N)) \
    --machine_rank=RANK --main_process_ip=MAIN_IP \
    --main_process_port=9999 --same_network \
    scripts/train_bar.py \
    config=configs/generator/bar_b_patch4.yaml \
    'dataset.params.train_shards_path_or_url=/path/to/imagenet-train-{000000..000320}.tar' \
    'dataset.params.eval_shards_path_or_url=/path/to/imagenet-val-{000000..000049}.tar' \
    training.per_gpu_batch_size=$((2048 / (8*N)))
```

## Evaluation on ImageNet-1K

We provide a [sampling script](./sample_imagenet.py) for reproducing generation results on ImageNet-1K benchmark.

### Setup Evaluation Tools
```bash
# Prepare ADM evaluation script
git clone https://github.com/openai/guided-diffusion.git

# Download reference statistics
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```

### Generate and Evaluate

**Example: BAR-B/4**
```bash
# Generate samples
torchrun --nnodes=1 --nproc-per-node=1 --rdzv-endpoint=localhost:9999 \
    sample_imagenet.py \
    config=configs/generator/bar_b_patch4.yaml \
    experiment.output_dir="bar_b_patch4" \
    experiment.generator_checkpoint=assets/generator/bar_b_patch4.bin \
    experiment.tokenizer_checkpoint=assets/tokenizer/bar_fsq_16bits.bin \
    model.generator.guidance_scale=9.6 \
    model.generator.mbm_head.randomize_temperature=1.4 \
    'model.generator.mbm_head.tokens_allocation=[64,64,64,64]'

# Evaluate FID
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    bar_b_patch4.npz

# Expected output:
# Inception Score: 274.70697021484375
# FID: 2.3366168749295753
# sFID: 6.311607318546635
# Precision: 0.79364
# Recall: 0.5984
```

**Example: BAR-B/2**
```bash
# Generate samples
torchrun --nnodes=1 --nproc-per-node=1 --rdzv-endpoint=localhost:9999 \
    sample_imagenet.py \
    config=configs/generator/bar_b_patch2.yaml \
    experiment.output_dir="bar_b_patch2" \
    experiment.generator_checkpoint=assets/generator/bar_b_patch2.bin \
    experiment.tokenizer_checkpoint=assets/tokenizer/bar_fsq_16bits.bin \
    model.generator.guidance_scale=5.5 \
    model.generator.mbm_head.randomize_temperature=2.0 \
    'model.generator.mbm_head.tokens_allocation=[16,16,16,16]'

# Evaluate FID
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    bar_b_patch2.npz

# Expected output:
# Inception Score: 293.40704345703125
# FID: 1.3484216683129944
# sFID: 4.931784360145002
# Precision: 0.7887
# Recall: 0.6377
```

**Example: BAR-B**
```bash
# Generate samples
torchrun --nnodes=1 --nproc-per-node=1 --rdzv-endpoint=localhost:9999 \
    sample_imagenet.py \
    config=configs/generator/bar_b.yaml \
    experiment.output_dir="bar_b" \
    experiment.generator_checkpoint=assets/generator/bar_b.bin \
    experiment.tokenizer_checkpoint=assets/tokenizer/bar_fsq_16bits_ft.bin \
    model.generator.guidance_scale=5.0 \
    model.generator.mbm_head.randomize_temperature=2.5 \
    'model.generator.mbm_head.tokens_allocation=[2,2,5,7]'

# Evaluate FID
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    bar_b.npz

# Expected output:
# Inception Score: 289.0171813964844
# FID: 1.1292903682207225
# sFID: 4.692096472506364
# Precision: 0.77354
# Recall: 0.6635
```

**Example: BAR-L**
```bash
# Generate samples
torchrun --nnodes=1 --nproc-per-node=1 --rdzv-endpoint=localhost:9999 \
    sample_imagenet.py \
    config=configs/generator/bar_l.yaml \
    experiment.output_dir="bar_l" \
    experiment.generator_checkpoint=assets/generator/bar_l.bin \
    experiment.tokenizer_checkpoint=assets/tokenizer/bar_fsq_16bits_ft.bin \
    model.generator.guidance_scale=5.3 \
    model.generator.mbm_head.randomize_temperature=3.0 \
    'model.generator.mbm_head.tokens_allocation=[2,2,5,7]'

# Evaluate FID
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    bar_l.npz

# Expected output:
# Inception Score: 296.94757080078125
# FID: 0.9926468757014959
# sFID: 4.673216174301388
# Precision: 0.7693
# Recall: 0.6858
```

**Example: BAR-L-res512**
```bash
# Generate samples
torchrun --nnodes=1 --nproc-per-node=1 --rdzv-endpoint=localhost:9999 \
    sample_imagenet.py \
    config=configs/generator/bar_l_res512.yaml \
    experiment.output_dir="bar_l_res512" \
    experiment.generator_checkpoint=assets/generator/bar_l_res512.bin \
    experiment.tokenizer_checkpoint=assets/tokenizer/bar_sfq_10bits_res512.bin \
    model.generator.guidance_scale=4.2 \
    model.generator.mbm_head.randomize_temperature=2.8 \
    'model.generator.mbm_head.tokens_allocation=[2,2,2,4]'

# Evaluate FID
python3 guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet512.npz \
    bar_l_res512.npz

# Expected output:
# Inception Score: 311.08929443359375
# FID: 1.0937443187164604
# sFID: 4.404387828127369
# Precision: 0.79574
# Recall: 0.644
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{yu2026autoregressive,
  title     = {Autoregressive Image Generation with Masked Bit Modeling},
  author    = {Yu, Qihang and Liu, Qihao and He, Ju and Zhang, Xinyang and Liu, Yang and Chen, Liang-Chieh and Chen, Xi},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
