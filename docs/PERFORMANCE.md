# Performance notes

What makes WAI Gen fast, what was measured, and what was tried and dropped. All numbers are from one
Apple M5 Max (128 GB), Release builds, unless noted. "Warm" = models already loaded.

## Where the time goes (current)

| Job | Time | Biggest part |
|---|---|---|
| WAI v16, 1024×1344, 28 steps | ~15-17 s | UNet sampling, ~0.6 s/step (MLX) |
| WAI v16 face repair (one face) | ~6 s (was 17.3 s) | 11 masked steps at 1024×1024 (MLX) |
| WAI-ANIMA, 1024×1344 → 1536×2016 hi-res + face repair, cold start | **157 s** (346 s this morning, 750 s originally) | the Anima model, now in MLX: 1.36 s/step at 1024×1344 |

## Round 1: engines (2026-11-19)

### SDXL (WAI-illustrious) in MLX instead of ComfyUI / stable-diffusion.cpp

WAI v16, 832×1216, 28 steps:

| Engine | Time |
|---|---|
| **MLX (WaiMLX, this app)** | **~12 s** |
| ComfyUI (PyTorch MPS) | 15.6 s |
| stable-diffusion.cpp (Metal) | 37.7 s |

The SDXL code is Apple's mlx-swift-examples StableDiffusion, with ComfyUI-style k-diffusion sampling, prompt
weighting and single-file checkpoint loading added. The cond/uncond passes run as one batch of 2, attention
is MLX's fused `scaledDotProductAttention`, and MLX 0.31 uses the M5's neural accelerators for matrix math.

The one fix that mattered: the VAE's middle attention originally built the full [HW × HW] score matrix
(~5 GB at 1248×1824), which pushed the second generation into swap (137 s). Fused attention, a bf16 VAE and
`Memory.cacheLimit = 2 GB` brought VAE decode from 7.6 s to 0.6 s and made repeated runs stable.

### WAI-ANIMA's VAE in MLX (`QwenImageVAE.swift`, `patches/stable-diffusion.cpp.patch`)

stable-diffusion.cpp runs the Qwen-Image (Wan 2.1) VAE through ggml's 3D convolutions, which are very slow on
Metal at these sizes:

| | stable-diffusion.cpp | MLX |
|---|---|---|
| decode 1024×1344 | ~45 s | **0.27 s** |
| decode 1536×2016 | ~135 s | **0.68 s** |
| encode 1536×2016 | ~85 s | **0.59 s** |

- For one frame the causal 3D VAE is exactly a 2D VAE (only the last temporal slice of each 3×3×3 kernel sees
  the image; `time_conv` is skipped on the first frame), so it runs as a plain 2D network.
- A 61-line patch adds `sd_set_vae_hook()` to stable-diffusion.cpp: every single-image VAE encode/decode of a
  context (txt2img, img2img, hi-res, face repair) goes to the app. The latent mean/std normalisation stays
  in stable-diffusion.cpp.
- Checked: encode → decode round trip 44.5-45 dB PSNR; a full txt2img where only the VAE differs is 53.9 dB
  from stable-diffusion.cpp's image; img2img 38.6 dB (one sampling step amplifies tiny differences).
- **Tried and dropped:** stable-diffusion.cpp's `vae_tiling` (154 tiles × 3.3 s, far slower) and
  `vae_conv_direct` (281 s vs 250 s for a 1-step img2img at 1536×2016).

### ESRGAN upscaler in MLX (`RealESRGAN.swift`, `TorchCheckpoint.swift`)

RealESRGAN_x4plus_anime_6B, 1024×1344 → 4096×5376: **1.97 s** in MLX vs 27.5 s in stable-diffusion.cpp
(60.6 dB PSNR between the two). This is the "upscaling" step of every hi-res image. `.pth` files are read by a
loader that implements only the pickle opcodes of a tensor `state_dict` and refuses any other callable (a
booby-trapped file calling `os.system` is rejected and nothing runs).

For WAI-ANIMA the app now runs the hi-res fix itself (base pass → MLX ESRGAN → img2img at the new size) instead
of stable-diffusion.cpp's built-in hi-res, so the upscaler and both VAE passes run in MLX.

**WAI-ANIMA, 1024×1344 → 1536×2016 with hi-res and face repair: 750 s → 346 s; face repair 110 s → 27 s.**

### Keeping models loaded

- The selected image model, the upscaler and the face-repair helpers load in the background when the app
  opens and when another model is picked, so the first image doesn't wait. Loading always yields to a waiting
  generation (each load step is queued behind it).
- They are freed after 5 minutes without generating (configurable) and reload on return.
- Optional: keep every image model loaded for instant switching. Measured footprint with WAI v16, v12 and
  WAI-ANIMA plus face repair loaded: **14 GB** (`footprint`).
- A cold WAI v16 start was only ~3.5 s slower than a warm one (macOS keeps the file cached), so converting
  checkpoints to a faster-loading format isn't worth it.

### Not speed, but found while measuring

Quitting after stable-diffusion.cpp had been used always crashed ("quit unexpectedly"): ggml's Metal backend
asserts in a C++ static destructor at `exit()` that every GPU buffer was freed. The app now frees all models
on `willTerminate`; the CLI leaves through `_exit`.

## Round 2 (2026-11-19 evening)

Ordered by effort: trivial and small changes first, then medium, then large.

### Trivial: 15 hi-res steps instead of 20 (all presets)

- WAI v16, same seed, 1024×1344 → 1536×2016, denoise 0.4: 20, 15 and 12 hi-res steps look the same at full size
  (12 loses a little texture in hair highlights). About 8 s saved per hi-res image.

  ![20 / 15 / 12 hi-res steps, WAI v16](images/hires-steps-wai-v16.jpg)
- WAI-ANIMA, same seed, 1.5× of a 1024×1344 image: 20 and 15 steps are indistinguishable; **214.7 s → 174.9 s**.

  ![20 / 15 hi-res steps, WAI-ANIMA](images/hires-steps-wai-anima.jpg)
- All presets now use 15; saved settings that still had the old default 20 are migrated once.

### Small changes

| Change | Measured | Kept? |
|---|---|---|
| `MLX.compile` on the UNet call (model passed as state, so LoRA updates are seen) | UNet step 0.628 → 0.609 s (batch 1), 0.571 → 0.538 s (batch 2) at 1024×1344; full image 17.23 → 16.95 s; **bit-identical output** | **yes** |
| WAI-ANIMA weights as `q8_0` (stable-diffusion.cpp `wtype`) | 2.72 → 2.65 s/step (−2.6 %), same 29 s total for 10 steps, output 30.8 dB from the original | no |
| Caching the text encoding between images | the MLX text encode takes 0.04 s | no (nothing to gain) |
| Generating "Images: 2" as one batch | UNet 0.609 → 0.538 s per image per step (−12 %), only for multi-image runs | not yet (would touch every stage: hi-res, face repair, previews) |

Notes:
- stable-diffusion.cpp's fast INT8 path (Metal "TensorOps", the M5's int8 units) is only used for ComfyUI-style
  pre-quantised `int8_tensorwise` safetensors, not for `wtype=q8_0`. Converting WAI-ANIMA to that format wasn't
  pursued because the large step below replaces stable-diffusion.cpp for WAI-ANIMA anyway.
- The benchmark lives in the CLI: `waigen-cli -m <checkpoint> -p x --bench-unet 1024x1344`.

### Medium: face repair for WAI v16 / v12 in MLX

Face repair used a second stable-diffusion.cpp context with the same SDXL checkpoint (~7 GB, ~20 s to load)
and took **17.3 s** warm per image with one face, longer than generating the image (16.4 s).

Now stable-diffusion.cpp only runs the YOLOv8 face detector (new `sd_adetailer_detect()` in
`patches/stable-diffusion.cpp.patch`, ~0.15-0.35 s), and the MLX engine that made the image repaints the faces.
`FaceRepair.swift` ports stable-diffusion.cpp's detailer step by step so the result matches:

- box → `dilate 4` (−1/+2 px) → 32 px padding → square crop around the centre → bilinear resize to 1024×1024;
- masked img2img at strength 0.4: the last `steps × strength` steps of the job's schedule, and after every step
  the prediction outside the mask is replaced by the original latent (`denoised·m + init·(1−m)`, mask
  downsampled nearest-max to latent size), exactly like stable-diffusion.cpp's denoise mask;
- paste back through a Gaussian-feathered mask (σ = 4, radius 10, edges clamped), bilinear resampling at
  half-pixel centres, as `composite_crop` does.

**17.3 s → 6.0 s per image with a face** (detection included), visually equivalent results, and no second copy of the checkpoint in memory.

![before / stable-diffusion.cpp / MLX face repair](images/face-repair-sdcpp-vs-mlx.jpg)

WAI-ANIMA keeps stable-diffusion.cpp's
own ADetailer (its repaint already uses the MLX VAE).

### Large: WAI-ANIMA's diffusion model in MLX (`AnimaDiT.swift`)

The Anima model (a Cosmos-Predict2-style DiT: 28 blocks, 2048 wide, 16 heads, plus a 6-block LLM adapter that
turns Qwen3-0.6B states into the context) is ported line by line from stable-diffusion.cpp's `anima.hpp`:
padding-mask channel and 2×2 patchify, `[cos | sin]` timestep embedding, AdaLN-Zero modulation
(`LN(x)·(1+scale)+shift`, gates), per-head RMS-normed q/k, rotate-half RoPE over (0, row, col) with 44/42/42
dims and NTK-scaled θ for rows and columns, tanh-GELU MLPs, `(ph pw c)` unpatchify.

stable-diffusion.cpp still does everything around it: Qwen3 text encoding, the sampler and CFG, the progress
and preview callbacks, face repair. The patch adds `sd_set_diffusion_hook()`: when set, Anima's forward pass
calls the app with the latent, timestep, text states and adapter tokens instead of running its ggml graph
(jobs with WAI-ANIMA LoRAs switch the hook off, because the LoRAs are applied inside stable-diffusion.cpp's model).
stable-diffusion.cpp loads weights lazily, so with the hook it never loads its own copy of the model.

| WAI-ANIMA, 1024×1344, 10 steps, same seed | stable-diffusion.cpp model | MLX model |
|---|---|---|
| per step | 2.72 s | **1.36 s** |
| total | 29.1 s | **14.0 s** |

The two images agree to 32.3 dB PSNR (bf16 rounding over 28 blocks × 10 steps; a wrong port gives ~10 dB)
and look the same:

![stable-diffusion.cpp vs MLX Anima model](images/anima-sdcpp-vs-mlx.jpg)

**In the app**, WAI-ANIMA 1024×1344 → 1536×2016 with hi-res (15 steps, MLX ESRGAN) and face repair, including
the cold model load: **156.8 s** (face repair 22.4 s), memory footprint 6.7 GB. The same job was 346 s this
morning and 750 s before the VAE port.

## Possible next steps

- Generate "Images: 2+" as one batch on MLX (−12 % per image, measured above).
- LoRAs for the MLX Anima model, so LoRA jobs don't fall back to stable-diffusion.cpp's model.
- Batch the cond/uncond passes of WAI-ANIMA into one MLX call (stable-diffusion.cpp calls the model twice per step).
