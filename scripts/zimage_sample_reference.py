#!/usr/bin/env python3
"""End-to-end Z-Image reference from the local ComfyUI: samples one image from the step-0 noise
(ref.safetensors "noise", 1024x1024) with a deterministic sampler (Euler), so WAI Gen's MLX pipeline can be
compared pixel for pixel on the same noise:

    /Users/cookies/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python3 scripts/zimage_sample_reference.py [scheduler]
    waigen-cli --zimage <aio> -p "<PROMPT>" -W 1024 -H 1024 --steps 9 --sampler euler --scheduler beta \
        --zimage-noise /Users/cookies/Developer/zimage-ref/ref.safetensors:noise -o mlx.png

Writes /Users/cookies/Developer/zimage-ref/comfy_euler_<scheduler>[_<lora>].png. Scheduler: beta (default) or simple.
Optional LoRA (ComfyUI's own loader): ... zimage_sample_reference.py beta /path/to/lora.safetensors 1.0
"""

import os
import sys

COMFY_DIR = "/Users/cookies/ComfyUI-Installs/ComfyUI/ComfyUI"
CKPT = "/Users/cookies/ComfyUI-Shared/models/checkpoints/z-image-anime-aio-bf16.safetensors"
REF = "/Users/cookies/Developer/zimage-ref/ref.safetensors"
OUT_DIR = "/Users/cookies/Developer/zimage-ref"
PROMPT = "anime, 1girl, solo, long hair, white shirt, denim jacket, smile, city street, day"
STEPS = 9

scheduler = sys.argv[1] if len(sys.argv) > 1 else "beta"
lora_path = sys.argv[2] if len(sys.argv) > 2 else None
lora_strength = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
sys.path.insert(0, COMFY_DIR)
sys.argv = [sys.argv[0]]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402


@torch.inference_mode()
def main():
    noise = load_file(REF)["noise"]  # [1, 16, 128, 128] float32
    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(CKPT, output_vae=True, output_clip=True)
    suffix = ""
    if lora_path:
        model, clip = comfy.sd.load_lora_for_models(model, clip, comfy.utils.load_torch_file(lora_path), lora_strength, 0)
        suffix = "_" + os.path.splitext(os.path.basename(lora_path))[0]
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    # ConditioningZeroOut, as the Turbo workflow's negative (unused at CFG 1)
    negative = [[torch.zeros_like(c[0]), {k: (torch.zeros_like(v) if torch.is_tensor(v) else v) for k, v in c[1].items()}]
                for c in positive]
    latent = torch.zeros_like(noise)
    samples = comfy.sample.sample(model, noise, STEPS, 1.0, "euler", scheduler, positive, negative, latent,
                                  denoise=1.0, seed=0)
    pixels = vae.decode(samples)  # [1, H, W, 3] in 0..1
    img = (pixels[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    out = os.path.join(OUT_DIR, "comfy_euler_{}{}.png".format(scheduler, suffix))
    Image.fromarray(img).save(out)
    print("sigmas:", [round(float(s), 5) for s in
                      comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, STEPS)])
    print("wrote", out)


if __name__ == "__main__":
    main()
