# Changelog

## 2026-11-20

- Removed the WAI v12 preset (the older version of WAI-illustrious-SDXL); WAI-ANIMA is now ⌘2.
- Fast mode (opt-in, inspector ▸ Image): DPM++ 2M Karras 16 steps on WAI v16/v12 (−32 %), Euler 16 steps on
  WAI-ANIMA (−28 %).
- WAI-ANIMA LoRAs run on the MLX model (no longer fall back to stable-diffusion.cpp's model): ~1.7× faster.
- Measured and dropped (see docs/PERFORMANCE.md, round 3): adapter caching, compiled Anima forward (opt-in only),
  quantized Anima weights, batched CFG passes, batched multi-image generation.

## 2026-11-19 (evening)

- Inspector: the model menu sits in the toolbar strip above it (no empty band), section titles are buttons
  (clicking the title collapses the section).
- WAI-ANIMA's diffusion model runs in MLX (2× faster per step); WAI-ANIMA hi-res + face repair 346 s → 157 s.
- Faster: face repair for WAI v16/v12 runs in MLX (17.3 s → 6.0 s per image), hi-res fix uses 15 steps
  (WAI-ANIMA hi-res 214.7 s → 174.9 s), the MLX UNet runs compiled. Details in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
- stable-diffusion.cpp patch now also exposes the face detector on its own (`sd_adetailer_detect`);
  both changes live in `patches/stable-diffusion.cpp.patch`.
- App icon.
- Models load in the background (selected model + upscaler + face repair), are freed after 5 idle minutes;
  optional "keep every image model loaded".
- Zoomable image, collapsible inspector sections (arrow or title), hideable prompt and gallery.
- Watermark / signature / Patreon tags in the default negatives.

## 2026-11-19

- First public version: SwiftUI app + CLI. WAI-illustrious SDXL in MLX, WAI-ANIMA through stable-diffusion.cpp
  with its VAE and the ESRGAN upscaler in MLX (WAI-ANIMA hi-res + face repair 750 s → 346 s).
- Fixed: "quit unexpectedly" after using stable-diffusion.cpp (ggml's exit-time assert).
