# Building Sketch to Photorealistic Image

This repository contains code, documentation, and model weights for a pipeline that converts hand-drawn architectural sketches into photorealistic renders, with strict structural fidelity.

---

## Model & Methodology Details

**Approach:**  
Extensive fine-tuning with LoRA adapters on Stable Diffusion XL (SDXL) combined with a ControlNet (scribble variant) backbone.  

**Base Models:**  
- **Stable Diffusion XL:** `stabilityai/stable-diffusion-xl-base-1.0`  
- **ControlNet Scribble:** `xinsir/controlnet-scribble-sdxl-1.0`  

**Methodology:**  
LoRA adapters were trained on paired datasets to enforce:
- Exact window and door placement
- Realistic proportions
- Accurate rendering of materials (limestone, cedar, etc.)

The pipeline leverages ControlNet conditioning to closely adhere to sketch contours, while LoRA fine-tuning improves stylistic consistency and structural precision.

---

## Data Strategy

**Dataset:**
- 20 paired sketch/reference image pairs collected and manually verified.

**Preprocessing:**
- Normalization to [-1,1]

**Augmentation:**
- **No augmentations were used in the final training run** (see Ablation Study below).

---

## Training Setup

**Environment:**
- Libraries: `torch`, `diffusers`, `peft`, `transformers`, `safetensors`
- Hardware: NVIDIA A100 GPU (40GB VRAM)

**Best Hyperparameters:**
- Batch Size: `1`
- Gradient Accumulation Steps: `8`
- Epochs: `30`
- Learning Rate: `1e-4`
- Gradient Clipping Norm: `1.0`
- Latent Scaling Factor: `0.18215`
- Scheduler: `DDPMScheduler`

**Training Notebook:**
https://colab.research.google.com/drive/1sVxUuKs8GLHS2Jcwxd7fBYowu18wAHa1?usp=sharing

---

## Ablation Study

Over **15 training runs** were conducted to evaluate the impact of augmentation strategies:

| Run Type                  | Window/Structure Accuracy | Visual Consistency |
|---------------------------|---------------------------|---------------------|
| No Augmentations          | ✅ Highest fidelity       | ✅ Consistent       |
| Random Rotations          | ❌ Slight distortion      | ❌ Inconsistent     |
| Random Flips              | ❌ Lower accuracy         | ❌ Inconsistent     |

Based on these experiments, **no augmentations** produced the most reliable structural adherence and visual realism.

---

## Training Loss

The following plot shows the training loss per epoch:

![Training Loss Curve](https://github.com/killtony/WHERENESS-Challenge/blob/main/training-loss-curve.png?raw=true)

The loss stabilized over ~30 epochs, confirming convergence.

---

## Inference

Hugging Face Space
An interactive demo is available here:
https://huggingface.co/spaces/arkane/whereness-challenge
*Note using CPU so inference takes several minutes

Inference prompt:
"ultra-realistic architectural rendering of a modern two-storey villa, white limestone panel façade and vertical cedar cladding, rooftop greenery, reflecting pool, overcast soft light, strictly matching the reference sketch, preserve exact window and door placement, no additional elements."
Negative prompt: "extra windows, distortion, lowres, watermark"

## Results

Sketch: 
![Generated image](https://github.com/killtony/WHERENESS-Challenge/blob/main/training-loss-curve.png?raw=true)

Fine-tuned generated image:
![Generated image](https://github.com/killtony/WHERENESS-Challenge/blob/main/training-loss-curve.png?raw=true)

Basemodel generated image:
![Generated image](https://github.com/killtony/WHERENESS-Challenge/blob/main/training-loss-curve.png?raw=true)

As you can see the Fine-tuned image preserves the exact window, door placement and overall structure of the building.
Note: the baseline model with the scribble controlnet already performs quite well, preserving all features. However, the fine-tuned image adds slightly more color, depth and realism which is a personal preference. 

---

## Further Study

Hugging Face Space
An interactive demo is available here:
https://huggingface.co/spaces/arkane/whereness-challenge
*Note using CPU so inference takes several minutes

## License
This repository is released under the MIT License.

## Acknowledgments
StabilityAI for SDXL
ControlNet authors for the scribble conditioning model
Hugging Face for hosting and serving models
