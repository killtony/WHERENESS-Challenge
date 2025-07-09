# Building Sketch to Photorealistic Image

This repository contains code, documentation and weights for a pipeline that converts hand-drawn architectural sketches into photorealistic renders.

## Model & Methodology Details

- **Approach:** Fine-tuning with LoRA adapters on Stable Diffusion XL + ControlNet.
- **Base Models:**
  - Stable Diffusion XL (`stabilityai/stable-diffusion-xl-base-1.0`)
  - ControlNet (scribble variant)
- **Methodology:**
  - LoRA adapters were trained on a dataset of paired sketches and photorealistic references.
  - Special care was taken to preserve exact window and door placement and to enforce material realism.

## Data Strategy

- **Dataset:**
  - Collected 20 paired sketch/reference images.
  - Preprocessing: resizing to 1024x1024, normalization to [-1,1].
  - Augmentation: random rotations and flips

## Training Setup

- **Environment:**
  - Libraries: `torch`, `diffusers`, `peft`, `transformers`, `safetensors`
  - Hardware: A100 GPU, 40GB VRAM.
- **Hyperparameters:**
  - Learning Rate: `1e-4`
  - Batch Size: `4`
  - Epochs: `20`
  - Latent scaling: `0.18215`
  - Scheduler: `DDPMScheduler`
- **Training Script:** [`train.py`](train.py)

## Inference

- **Inference Script:** [`inference.py`](inference.py)
- **Example Usage:**
  ```bash
  python inference.py --sketch_path examples/example_sketch.png --output_path outputs/generated.png
