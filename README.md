# WAI Gen

A native macOS app (SwiftUI) for generating anime images with the **WAI** model family. It runs
the models itself: no Python, no ComfyUI server, no Electron. It does what
[Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app) does on top of
ComfyUI (character picker, tag autocomplete, hi-res fix, face repair, AI prompt), as one app.

| Model | Engine |
|---|---|
| WAI-illustrious-SDXL v16, v12 | **MLX** (Apple's ML framework), SDXL implemented in Swift |
| WAI-ANIMA v1.0 | **stable-diffusion.cpp** (Metal) for the diffusion model and text encoder, **MLX** for the VAE and the upscaler |

Picking a model applies its recommended settings (steps, CFG, sampler, size, hi-res, negatives).

## Features

- Character sidebar with the 5000+ characters and thumbnails of Character Select SAA, favourites, search
- Danbooru tag autocomplete, `(tag:1.2)` weights, wildcards (`__name__`, nested)
- Hi-res fix with an ESRGAN upscaler (e.g. RealESRGAN x4plus anime 6B), face repair (ADetailer-style, YOLOv8 face detector)
- Default negatives include watermark / signature / Patreon tags, which these models otherwise like to draw
- LoRAs for both engines, with a warning when a LoRA was made for the other model family
- Live preview of every step (latent projection, no extra time)
- Gallery: upscale with a detail pass or upscaler-only ×2, reuse prompt/settings/seed, info, ← / → to browse
- Zoomable image (pinch, ⌘+ / ⌘−, double-click for 100 %, ⌘0 / ⌘9), collapsible inspector sections,
  hideable prompt and gallery (⌥⌘P / ⌥⌘G)
- Models load in the background when the app opens (the selected model with its upscaler and face repair,
  or every image model if you want instant switching) and are freed after 5 minutes idle
- AI prompt: turn a sentence into tags with any OpenAI-compatible local server (MLX-Serve, LM Studio, ...)
- PNGs carry A1111-style `parameters` text (readable by other tools) plus the full settings as JSON
- `waigen-cli` for scripting and benchmarks

## Speed (Apple M5 Max)

| | WAI Gen | Before |
|---|---|---|
| WAI v16, 832×1216, 28 steps | **~12 s** (MLX) | 15.6 s ComfyUI, 37.7 s stable-diffusion.cpp |
| WAI-ANIMA VAE decode, 1024×1344 | **0.27 s** (MLX) | ~45 s (stable-diffusion.cpp Metal) |
| WAI-ANIMA VAE decode / encode, 1536×2016 | **0.68 s / 0.59 s** | ~135 s / ~85 s |
| RealESRGAN x4plus anime 6B, 1024×1344 → 4096×5376 | **1.97 s** (MLX) | 27.5 s (stable-diffusion.cpp) |

The MLX VAE and upscaler match stable-diffusion.cpp's output (PSNR 53.9 dB and 60.6 dB). They are also
published on their own: [mlx-swift-qwen-image-vae](https://github.com/ramirobuisan/mlx-swift-qwen-image-vae)
and [mlx-swift-realesrgan](https://github.com/ramirobuisan/mlx-swift-realesrgan).

## Build

Requirements: Apple silicon, macOS 14+, Xcode 16+, CMake.

```bash
scripts/build-sdcpp.sh          # stable-diffusion.cpp at the tested commit + patches/sdcpp-vae-hook.patch
scripts/make-app.sh --install   # Release build -> ~/Applications/WAI Gen.app
```

Open `Package.swift` in Xcode once and trust the mlx-swift build plugin if it asks (xcodebuild needs it
for the Metal shaders).

## Models

Point **Settings → Models** at a folder with the ComfyUI layout (`~/ComfyUI-Shared/models` or
`~/ComfyUI/models` are found automatically):

| File | Where |
|---|---|
| `checkpoints/waiIllustriousSDXL_v160.safetensors` (and/or v12) | [WAI-illustrious-SDXL on Civitai](https://civitai.com/models/827184) |
| `diffusion_models/waiANIMA_v10Base10.safetensors` | WAI-ANIMA on Civitai |
| `text_encoders/qwen_3_06b_base.safetensors`, `vae/qwen_image_vae.safetensors` | [Comfy-Org](https://huggingface.co/Comfy-Org) repackaged files |
| `upscale_models/RealESRGAN_x4plus_anime_6B.pth` | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) |
| `detectors/face_yolov8m.safetensors` | `face_yolov8m.pt` from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer), converted with `scripts/convert_yolov8_safe.py` (no pickle code is run) |
| `loras/*.safetensors` | your LoRAs |

Character data, thumbnails and the tag list are downloaded on first start from Character Select SAA's
data (or copied from an existing SAA install).

## How the WAI-ANIMA speed-up works

stable-diffusion.cpp runs WAI-ANIMA's Qwen-Image (Wan 2.1) VAE through ggml's 3D convolutions, which
take minutes at hi-res sizes on Metal. `patches/sdcpp-vae-hook.patch` adds `sd_set_vae_hook()`: when an
app registers it, every single-image VAE encode/decode in that context (txt2img, img2img, hi-res, face
repair) calls the app instead, and stable-diffusion.cpp keeps doing the latent normalisation. WAI Gen
answers with an MLX implementation of the VAE. For one frame the causal 3D VAE is exactly a 2D VAE
(only the last temporal slice of each kernel sees the image), so it runs as fast as an SDXL VAE.

The ESRGAN upscaler runs in MLX too. `.pth` files are read by a small loader that only accepts plain
tensor dictionaries and never executes pickle code.

## Credits

- [mlx-swift](https://github.com/ml-explore/mlx-swift) and the StableDiffusion code of
  [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (MIT, Apple; see `THIRD_PARTY_LICENSE_mlx-swift-examples.txt`)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT)
- [Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app) (MIT) for the character data and the workflow this app reproduces
- Sampling and checkpoint-key conventions follow [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and k-diffusion; CLIP BPE vocabulary from OpenAI CLIP (MIT)
- The WAI models are by WAI0731; check each model's license on Civitai before using the images

## License

MIT
