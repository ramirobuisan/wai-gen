# Changelog

## 0.3.1 (2026-11-20)

- Hover over a style's or a character's thumbnail to see it bigger: after a moment a card with the image, the name and
  its Danbooru count appears over the canvas beside the sidebar (moving to another thumbnail swaps it at once).

Update: WAI Gen 0.2.0 or later offers this version in **Settings → Updates** (and an "Update to 0.3.1" button in the
toolbar). From an older build, download WAI-Gen.zip below, move **WAI Gen.app** to Applications and, the first time,
allow it in **System Settings → Privacy & Security → Open Anyway**.

## 0.3.0 (2026-11-20)

- Styles in the sidebar (Characters | Styles): 16,006 Danbooru artist styles from ThetaCursed's Illustrious & NoobAI
  Style Explorer (MIT), each with a sample made with Illustrious XL from one fixed prompt. Search, favourites, sort by
  popularity (Danbooru works), name or distinctiveness, or shuffle. A click adds the artist after the characters
  (`@name` on WAI-ANIMA, as Anima's model card asks); the chosen styles show under the prompt, are saved in each
  image's record and come back with "use settings". The list (2.3 MB) and each sample (~55 KB) are downloaded when
  first needed, from a pinned commit, and kept in Application Support.
- Characters sort by popularity (Danbooru posts, from the tag list, shown on each row) or by name.
- Name sorting ignores leading punctuation; "1 match" instead of "1 matches".

Update: WAI Gen 0.2.0 offers this version in **Settings → Updates** (and an "Update to 0.3.0" button in the toolbar).
From an older build, download WAI-Gen.zip below, move **WAI Gen.app** to Applications and, the first time, allow it
in **System Settings → Privacy & Security → Open Anyway**.

## 0.2.0 (2026-11-20)

The first release on GitHub. From this version on, WAI Gen updates itself from the releases here.

- Updates: **Settings → Updates** and **WAI Gen → Check for Updates…**. Stable channel (releases) or Beta
  (pre-releases too); checks once a day when automatic checks are on and shows an "Update to x.y.z" button in the
  toolbar. The download's size and SHA-256 must match GitHub's, it must be WAI Gen at that version with an intact
  signature; then it replaces the app in place and reopens it. Models, images and settings stay.
- Generating on another Mac now waits while macOS asks for Local Network access and explains a blocked or
  unreachable Mac (with a button for the Local Network settings).
- Everything in the sections below: WAI v16 and WAI-ANIMA in Swift + MLX, hi-res fix, face repair, LoRAs,
  upscalers, live preview, step cache, and generating on another Mac.

Install: unzip, move **WAI Gen.app** to Applications, and the first time allow it in **System Settings → Privacy &
Security → Open Anyway** (the app isn't notarized). Later updates install from the app without that step.

## 2026-11-20 (early morning)

- Generate on another Mac: the laptop now waits (15 s) while macOS asks for Local Network access instead of
  failing at once with "Network is down", and a blocked or unreachable Mac gets an error that says what to do
  (with a button for Local Network settings).
- Generate on another Mac: Settings → Share this Mac turns a Mac into a generator (WebSocket server, Bonjour,
  pairing code, local-network and Tailscale addresses only); Settings → Generator makes a laptop use it (models,
  LoRAs and upscalers listed from it, live preview, images saved locally). Per-job cancel in the engine queue so
  one Mac's Stop doesn't stop the other's job. CLI: `--remote`, `--code`, `--preset`, `--serve-test`.

## 2026-11-20 (late night)

- WAI v16: the UNet's deepest-level 3×3 convolutions run as nine matrix multiplies, which use the M5 GPU's neural
  accelerators (MLX's conv2d doesn't): 3× faster for those layers, 6-10 % per image, same picture.
  `WAIGEN_CONV_MATMUL=0` turns it off; `waigen-cli --bench-conv` measures every convolution size (round 7).

## 2026-11-20 (night)

- stable-diffusion.cpp removed: WAI-ANIMA runs end to end in Swift + MLX (`WaiAnima`: Philox noise, flow
  schedules, Euler / Euler a / DPM++ 2M, CFG, img2img, masked face repaint, EasyCache; `AnimaText`: prompt
  weights, Qwen2 BPE and T5 unigram tokenizers). Ported line by line and checked against it: identical tokens,
  a bit-identical 1-step image, the same pictures for the same seeds. No CMake step, no patch; the app is 22 MB
  (was 61 MB).
- WAI-ANIMA LoRAs: LoHa and LoKr are merged into the MLX model too (stable-diffusion.cpp crashed on LoKr).
- Upscalers the app can't run (e.g. 4x-UltraSharpV2, a DAT model) are hidden from the picker; a saved choice
  falls back to one that runs.
- WaiAnima caps MLX's buffer cache at 2 GB: a hi-res job peaks at 16 GB instead of 34 GB, same speed.
- CLI is MLX-only (`--detect`, `--img2img`, `--detail-only`, `--step-cache` for both models, `--make-test-lycoris`).

## 2026-11-20 (late)

- The face detector (YOLOv8) runs in MLX for both models: same boxes, 0.011 s instead of 0.07-0.27 s. WAI-ANIMA's
  face repair is driven from Swift (MLX detector, ADetailer geometry, stable-diffusion.cpp's masked img2img per
  crop): same result (52.6 dB), one detection instead of two. WAI v16 no longer uses stable-diffusion.cpp.
- WAI-ANIMA's Qwen3 text encoder runs in MLX through a new hook in the patch (`sd_set_text_encoder_hook`);
  stable-diffusion.cpp keeps the tokenizers and no longer loads its copy: first prompt 0.33 s → 0.08 s.
- `WAIGEN_SDCPP_LOG=info` prints stable-diffusion.cpp's stage timings. CLI: `--detect-test`, `--detect-engine`,
  `--te-engine`, `--bench-te`, `WAIGEN_DUMP_CONTEXT`.

## 2026-11-20 (night)

- Step cache (opt-in, inspector ▸ Image): EasyCache step skipping for both models. WAI-ANIMA uses
  stable-diffusion.cpp's `easycache` (threshold 0.5, 8 of 25 model calls skipped, −26 to −32 %); WAI v16 gets the
  same rule in the MLX sampler (base, hi-res and face repair; full job −26 %). Recorded in the image parameters.
- WAI-ANIMA face repair repaints at 768 px unless a face crop is bigger (15.9 s → 8.7 s per face, same look),
  and skips the pass when no face is found.
- Measured and dropped (docs/PERFORMANCE.md, round 4): `spectrum` cache (hazy images), `taylorseer`/`dbcache`
  (don't apply to the MLX model), fp16 weights for the MLX Anima model (no gain).
- CLI: `--cache`, `--step-cache`, `--face-size`, `--face-steps`, `--face-cache`, `--detail-only`, `--anima-dtype`.

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
