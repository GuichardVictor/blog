---
title: "LoRA finetuning"
date: 2023-02-03T16:24:50+01:00
draft: false
tags:
  - ml
  - research
categories:
  - programming
---

LoRAs are a recent approach in finetuning LLM and Diffusion models.
They work by injecting new weights to specific layers,
usually at the attention layers as they are some of the most crutial parts of these models.

They allow both efficient fine tuning and lower file size for sharing fine tuned models.

Diffusion models fine tuning exists in many forms, the most frequent are Dreambooth, textual inversion and LoRA. The latter is the most common form as they allow for an easy and fast finetuning and is compatible with the Dreambooth approach.

<!--more-->


## LoRA Dreambooth


<ins>@cloneofsimo</ins> was the first to introduce this concept to Stable Diffusion. In this blog, I will use his [repository](https://github.com/cloneofsimo/lora) to showcase an example of training and inference a new object concept.

### Training

While training a diffusion model can be costly, training a LoRA is very efficient as it only requires to train the LoRA weights while freezing the others.

LoRA Dreambooth allow to learn new concept in both the object and style form. We will focus on the object form, but the training and inference approach is exactly the saame for the style form.

Using Huggingface efforts to accelerate and simplify training we can easily launch the process with a single command and without the need of a extremly powerful gpu.

This is how you'd use it to fine-tune a model using pictures of __my dog__.


```sh
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="./data/mug"
export OUTPUT_DIR="./trained_models/"
export TEMPLATE="object"
export TOKEN="<mydog>"
export DEVICE="cuda:0"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens=$TOKEN \
  --use_template=$TEMPLATE \
  --save_steps=100 \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device=$DEVICE \
  --lora_rank=1 \
```

Running this using on an rtx 2080 ti should take around 15 minutes and will generate multiple checkpoint of the trained LoRA.

### Inference

To infer with trained LoRA, we need to patch the stable diffusion model used. Basically, we load the LoRA weights, and add them to the required layers. We can then scale the LoRA weights using an alpha parameter that will follow this formula:

$W_{sd} = W_{StableDiffusion} + \alpha W_{LoRA} $

By setting $\alpha$ at 0, we will only use the weights of Stable Diffusion, setting at 1 will make it fully use the LoRA weights, and greater than 1 will make the model mainly use the LoRA weights.

The inference code is as simple as this:
```py
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from lora_diffusion import tune_lora_scale, patch_pipe
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--prompt")
    parser.add_argument("--scale", type=float, default=0.8)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # Load Stable Diffusion
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    # Apply the LoRA layers on the model
    patch_pipe(
        pipe,
        args.model,
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    # Tune the LoRA alpha
    tune_lora_scale(pipe.unet, args.scale)
    tune_lora_scale(pipe.text_encoder, args.scale)

    # Inference
    torch.manual_seed(0)
    image = pipe(
        args.prompt,
        num_inference_steps=50,
        guidance_scale=7
    ).images[0]
    image.save("output.jpg")
```

Here is an image of my cute dog:

<p align="center">
  <img src="/posts/2023-06-17/images/my_dog.png" /></br>
  <em>My cute dog :D</em>
</p>

And here are some generated images:

<p align="center">
  <img src="/posts/2023-06-17/images/generated_1.png" />
    <img src="/posts/2023-06-17/images/generated_2.png" /></br>
  <em>< dog > sitting at the beach</em>
</p>

### Side Notes

A recent paper of Google claims to provide much better results in the styling category using approach called (__StyleDrop__)[https://styledrop.github.io/].
Combining __StyleDrop__ for the style and __Dreambooth__ to learn new objects might allow to generate images with even more accurate results for more specific use cases.