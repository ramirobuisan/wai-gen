#!/usr/bin/env python3
"""Step 0 of docs/plans/z-image-anime.md: dump reference tensors for the Z-Image
Turbo port from the local ComfyUI implementation.

Run with ComfyUI's own venv python, from the ComfyUI folder:
    /Users/cookies/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python3 \
        /Users/cookies/Developer/wai-gen-zimage/scripts/zimage_reference.py

Writes /Users/cookies/Developer/zimage-ref/ref.safetensors (all float32) and
logs shapes, dtypes and min/max/mean of every tensor.
"""

import os
import sys

COMFY_DIR = "/Users/cookies/ComfyUI-Installs/ComfyUI/ComfyUI"
CKPT = "/Users/cookies/ComfyUI-Shared/models/checkpoints/z-image-anime-aio-bf16.safetensors"
OUT_DIR = "/Users/cookies/Developer/zimage-ref"
OUT = os.path.join(OUT_DIR, "ref.safetensors")
PROMPT = "anime, 1girl, solo, long hair, white shirt, denim jacket, smile, city street, day"
SEED = 42

sys.path.insert(0, COMFY_DIR)
sys.argv = [sys.argv[0]]  # comfy.cli_args parses argv on import

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402


def stats(name, t):
    t32 = t.detach().to(torch.float32)
    print("  {:24s} shape={!s:24s} dtype={:14s} min={:+.5f} max={:+.5f} mean={:+.5f}".format(
        name, tuple(t.shape), str(t.dtype), t32.min().item(), t32.max().item(), t32.mean().item()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Deterministic noise latent for a 1024x1024 image (latent 128x128x16),
    # in "sampler space" (i.e. what process_latent_in already produced).
    torch.manual_seed(SEED)
    noise = torch.randn(1, 16, 128, 128, dtype=torch.float32)  # cpu, float32
    # and one for 832x1216 (latent 152x104: 76x52 = 3952 image tokens, padded to 3968 with x_pad_token)
    noise_tall = torch.randn(1, 16, 152, 104, dtype=torch.float32)

    print("Loading checkpoint (this reads 20.5 GB)...")
    model_patcher, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
        CKPT, output_vae=True, output_clip=True)
    print("model_config:", model_patcher.model.model_config)
    print("unet_config:", model_patcher.model.model_config.unet_config)
    print("sampling_settings:", model_patcher.model.model_config.sampling_settings)

    # Materialise + move the DiT to its load device.
    mm.load_models_gpu([model_patcher], force_full_load=True)
    model = model_patcher.model
    device = model_patcher.load_device
    dtype = model.get_dtype_inference()
    print("diffusion device:", device, "dtype:", dtype, "torch:", torch.__version__)

    out = {"noise": noise, "noise_832x1216": noise_tall}

    # --- text encoder -----------------------------------------------------
    tokens = clip.tokenize(PROMPT)
    tks = tokens["qwen3_4b"][0]  # one batch of (token, weight) pairs
    token_ids = [t for (t, w) in tks]
    tok_mask = [w for (t, w) in tks]
    print("num tokens:", len(token_ids))
    print("token ids:", token_ids)
    out["tokens"] = torch.tensor(token_ids, dtype=torch.int64)
    out["token_mask"] = torch.tensor(tok_mask, dtype=torch.int32)

    enc = clip.encode_from_tokens(tokens, return_dict=True)
    cond = enc["cond"]
    te_mask = enc.get("attention_mask", None)
    print("cond device/dtype:", cond.device, cond.dtype)
    stats("cond", cond)
    if te_mask is not None:
        stats("te_attention_mask", te_mask)
        num_tokens = int(max(1, te_mask.sum().item()))
        out["te_attention_mask"] = te_mask.to(torch.float32)
    else:
        num_tokens = max(1, int(sum(tok_mask)))
    out["num_tokens"] = torch.tensor([num_tokens], dtype=torch.int64)
    print("num_tokens passed to the model:", num_tokens)
    out["cond"] = cond

    # --- DiT ---------------------------------------------------------------
    x = noise.to(device=device, dtype=dtype)
    context = cond.to(device=device, dtype=dtype)
    for sigma in (1.0, 0.5):
        timesteps = torch.tensor([sigma], device=device, dtype=torch.float32)
        with torch.no_grad():
            dit_out = model.diffusion_model(x, timesteps, context=context, num_tokens=num_tokens)
        stats("dit_out_s{:.1f}".format(sigma), dit_out)
        out["dit_out_s{:.1f}".format(sigma)] = dit_out

        # What the sampler actually sees through apply_model (CONST flow:
        # calculate_denoised = model_input - sigma * model_output).
        with torch.no_grad():
            ap_out = model.apply_model(x, torch.tensor([sigma], device=device, dtype=torch.float32),
                                       c_crossattn=context, num_tokens=num_tokens)
        stats("apply_model_s{:.1f}".format(sigma), ap_out)
        out["apply_model_s{:.1f}".format(sigma)] = ap_out
        denoised = x.float() - sigma * dit_out.float()
        d = (ap_out.float() - denoised).abs().max().item()
        print("  apply_model vs (x - sigma*dit_out) max abs diff:", d)
        stats("denoised_s{:.1f}".format(sigma), denoised)
        out["denoised_s{:.1f}".format(sigma)] = denoised

    # 832x1216 at sigma 0.5: exercises the image-token padding (x_pad_token) and the crop after unpatchify
    with torch.no_grad():
        dit_tall = model.diffusion_model(noise_tall.to(device=device, dtype=dtype),
                                         torch.tensor([0.5], device=device, dtype=torch.float32),
                                         context=context, num_tokens=num_tokens)
    stats("dit_out_832x1216_s0.5", dit_tall)
    out["dit_out_832x1216_s0.5"] = dit_tall

    # --- VAE ---------------------------------------------------------------
    crop = noise[:, :, :64, :64]  # raw latent crop treated as a z (vae.decode does not rescale)
    vae_dtype = vae.vae_dtype
    with torch.no_grad():
        pixels = vae.decode(crop.to(device=device, dtype=vae_dtype))
    stats("vae_decode_crop64", pixels)
    out["vae_decode_crop64"] = pixels

    # --- save ---------------------------------------------------------------
    out = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in out.items()}
    save_file(out, OUT, metadata={"prompt": PROMPT, "seed": str(SEED)})
    print("wrote", OUT)
    print("all keys:", sorted(out.keys()))


if __name__ == "__main__":
    main()
