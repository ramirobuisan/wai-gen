# Changelog

## 2026-11-19 (evening)

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
