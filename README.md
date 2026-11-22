<img src="Resources/AppIcon-1024.png" width="128" alt="WAI Gen icon">

# WAI Gen

A native macOS app (SwiftUI) for generating anime images with the **WAI** model family. It runs
the models itself, entirely in [MLX](https://github.com/ml-explore/mlx-swift): no Python, no ComfyUI server,
no Electron, no C++ inference library. It does what
[Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app) does on top of
ComfyUI (character picker, tag autocomplete, hi-res fix, face repair, AI prompt), as one app.

| Model | Engine |
|---|---|
| WAI-illustrious-SDXL v16 | **MLX** (Apple's ML framework), SDXL implemented in Swift |
| WAI-ANIMA v1.0 | **MLX**: the Anima diffusion model, Qwen3 text encoder, Qwen-Image VAE, tokenizers and flow sampler ported from stable-diffusion.cpp (same seeds give the same images) |

The ESRGAN upscaler and the YOLOv8 face detector run in MLX too.

Picking a model applies its recommended settings (steps, CFG, sampler, size, hi-res, negatives).

## Features

- Character sidebar with the 5000+ characters and thumbnails of Character Select SAA, favourites, search, sorted by
  popularity (Danbooru posts) or name
- Styles sidebar: 16,006 Danbooru artist styles, each with a sample from the same prompt so only the style changes
  ([ThetaCursed's Style Explorer](https://github.com/ThetaCursed/Illustrious-NoobAI-Style-Explorer), made with
  Illustrious XL, the base of WAI v16); sort by popularity, name or distinctiveness, or shuffle; a click adds the
  artist to the prompt (as `@name` on WAI-ANIMA, as Anima expects)
- Hover over a character or style thumbnail to see it bigger
- Danbooru tag autocomplete, `(tag:1.2)` weights, wildcards (`__name__`, nested)
- Hi-res fix with an ESRGAN upscaler (e.g. RealESRGAN x4plus anime 6B), face repair (ADetailer-style, YOLOv8 face detector)
- Default negatives include watermark / signature / Patreon tags, which these models otherwise like to draw
- LoRAs for both engines (merged into the MLX models), with a warning when a LoRA was made for the other model family
- Fast mode: a quicker sampler with fewer steps (~30 % faster, slightly different look)
- Step cache: skips model calls on steps that barely change the image (EasyCache, ~25-30 % faster, nearly the same image)
- Faster hi-res guidance: guidance on every 2nd (WAI v16) or 3rd (WAI-ANIMA) hi-res step, reused in between (~10-18 % faster, nearly the same image)
- Live preview of every step (latent projection, no extra time)
- Gallery: upscale with a detail pass or upscaler-only ×2, reuse prompt/settings/seed, info, ← / → to browse,
  ⌘⌫ or the trash button to move an image to the Trash (⌘Z puts it back), Copy Image (⇧⌘C) copies the PNG,
  thumbnails drag out to Finder
- Zoomable image (pinch, ⌘+ / ⌘−, double-click for 100 %, ⌘0 / ⌘9), collapsible inspector sections,
  hideable prompt and gallery (⌥⌘P / ⌥⌘G)
- Models load in the background when the app opens (the selected model with its upscaler and face repair,
  or every image model if you want instant switching) and are freed after 5 minutes idle
- AI prompt: turn a sentence into tags with any OpenAI-compatible local server (MLX-Serve, LM Studio, ...)
- PNGs carry A1111-style `parameters` text (readable by other tools) plus the full settings as JSON
- Generate on another Mac: a laptop without the models uses a Mac that has them, on the same Wi-Fi or over Tailscale
- `waigen-cli` for scripting and benchmarks

## Speed (Apple M5 Max)

| | WAI Gen | Before |
|---|---|---|
| WAI v16, 832×1216, 28 steps | **~12 s** (MLX) | 15.6 s ComfyUI, 37.7 s stable-diffusion.cpp |
| WAI-ANIMA VAE decode, 1024×1344 | **0.27 s** (MLX) | ~45 s (stable-diffusion.cpp Metal) |
| WAI-ANIMA VAE decode / encode, 1536×2016 | **0.68 s / 0.59 s** | ~135 s / ~85 s |
| RealESRGAN x4plus anime 6B, 1024×1344 → 4096×5376 | **1.97 s** (MLX) | 27.5 s (stable-diffusion.cpp) |
| WAI-ANIMA model, 1024×1344, per step | **1.36 s** (MLX) | 2.72 s (stable-diffusion.cpp) |
| WAI v16 face repair, one face | **~3 s** (MLX, 768 px for faces up to that size) | 17.3 s (stable-diffusion.cpp) |
| WAI v16 1024×1344 → 1536×2016, hi-res + face repair, in the app, Step cache on | **31.6 s** | |
| WAI-ANIMA face repair, one face | **~5-9 s** (768 px repaint) | ~16 s at 1024 px |
| Face detector (YOLOv8m), warm | **0.011 s** (MLX) | 0.068-0.27 s (stable-diffusion.cpp) |
| WAI-ANIMA text encoder, first prompt / later prompts | **0.08 s / 0.01 s** (MLX) | 0.33 s / 0.02 s |
| WAI-ANIMA 1024×1344 → 1536×2016, hi-res + face repair, in the app, Step cache on | **76 s** | 157 s before the step cache, 750 s with stable-diffusion.cpp alone |
| App size | **22 MB** | 61 MB with stable-diffusion.cpp linked in |

The MLX VAE and upscaler match stable-diffusion.cpp's output (PSNR 53.9 dB and 60.6 dB). They are also
published on their own: [mlx-swift-qwen-image-vae](https://github.com/ramirobuisan/mlx-swift-qwen-image-vae)
and [mlx-swift-realesrgan](https://github.com/ramirobuisan/mlx-swift-realesrgan).

More numbers, including what was tried and dropped: [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## Install and update

Download `WAI-Gen.zip` from [Releases](https://github.com/ramirobuisan/wai-gen/releases), unzip it and move
**WAI Gen.app** to Applications (or ~/Applications). The app isn't notarized: the first time, macOS refuses to open
it; go to **System Settings → Privacy & Security** and press **Open Anyway** (once).

After that it updates itself from this repository's releases: **Settings → Updates** (or **WAI Gen → Check for
Updates…**). The **Stable** channel follows releases; **Beta** also takes pre-releases. With automatic checks on
(the default) it looks once a day and shows an **Update to x.y.z** button in the toolbar. Installing downloads the
release's `WAI-Gen.zip`, checks its size and SHA-256 against GitHub's, that it is WAI Gen at that version and that
its signature is intact, swaps it in place and reopens the app. Models, images and settings live outside the app
and stay. It waits for queued images (and other Macs' jobs) to finish first.

Releasing (maintainers): bump `VERSION`, add a `## <version>` section at the top of `CHANGELOG.md` (it becomes
the release notes), commit and push, then `scripts/release.sh` (or `scripts/release.sh --beta` for a
pre-release). It builds, zips, tags `v<version>` and creates the GitHub release with `gh`.

## Build

Requirements: Apple silicon, macOS 14+, Xcode 16+.

```bash
scripts/make-app.sh --install   # Release build -> ~/Applications/WAI Gen.app
swift scripts/make-icon.swift   # only to redraw the app icon (Resources/AppIcon.icns)
```

Open `Package.swift` in Xcode once and trust the mlx-swift build plugin if it asks (xcodebuild needs it
for the Metal shaders).

## Models

Point **Settings → Models** at a folder with the ComfyUI layout (`~/ComfyUI-Shared/models` or
`~/ComfyUI/models` are found automatically):

| File | Where |
|---|---|
| `checkpoints/waiIllustriousSDXL_v160.safetensors` | [WAI-illustrious-SDXL on Civitai](https://civitai.com/models/827184) |
| `diffusion_models/waiANIMA_v10Base10.safetensors` | WAI-ANIMA on Civitai |
| `text_encoders/qwen_3_06b_base.safetensors`, `vae/qwen_image_vae.safetensors` | [Comfy-Org](https://huggingface.co/Comfy-Org) repackaged files |
| `upscale_models/RealESRGAN_x4plus_anime_6B.pth` | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) |
| `detectors/face_yolov8m.safetensors` | `face_yolov8m.pt` from [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer), converted with `scripts/convert_yolov8_safe.py` (no pickle code is run) |
| `loras/*.safetensors` | your LoRAs |

Character data, thumbnails and the tag list are downloaded on first start from Character Select SAA's
data (or copied from an existing SAA install). The style list (2.3 MB) is downloaded the first time you open
Styles, and each style's sample (about 55 KB) the first time it's shown; they're kept in
`~/Library/Application Support/WAI Gen/data/styles` (delete that folder to free the space).

## Generating on another Mac

WAI Gen on a laptop can use another Mac's GPU and models, e.g. a MacBook Air generating on a MacBook Pro:

1. On the Mac with the models: **Settings → Share this Mac → Let other Macs generate on this Mac**. It shows a
   pairing code and its addresses (Wi-Fi and Tailscale, port 47860). Keep WAI Gen open there.
2. On the laptop: **Settings → Generator → Generate on**: pick that Mac (found automatically on the same network)
   or "Another address" and type its Tailscale name (`macbook-pro.your-tailnet.ts.net`) or 100.x address; enter
   the pairing code and press "Use this Mac".

The laptop then shows the other Mac's models, LoRAs and upscalers, the live preview streams back, and the images
are saved in the laptop's gallery with their settings. The laptop needs no model files (and doesn't load any).

If the laptop says "Network is down" or "No route to host", macOS is blocking WAI Gen from the local network: allow
it when macOS asks, or turn it on in **System Settings → Privacy & Security → Local Network** (the error has a
button for it) and reopen WAI Gen. Every new build asks again. The Tailscale address doesn't need that permission.

Protocol: one WebSocket per job (`Sources/WaiRemote`, Network.framework, Bonjour `_waigen._tcp`). The sharing Mac
accepts only local-network and Tailscale addresses, only after the pairing code (wrong codes are slowed down), and
only references to files in its own models folder. Over Wi-Fi the traffic isn't encrypted; over Tailscale it is.
Jobs from other Macs queue with the Mac's own jobs and use the models already loaded; Stop on one Mac only stops
that Mac's job. `waigen-cli --remote host[:port] --code ABC-123 -p "..."` (or `--remote bonjour:<name>`) is a
command-line client.

## How WAI-ANIMA runs in MLX

WAI-ANIMA first ran through stable-diffusion.cpp. Piece by piece it moved to MLX, each piece checked against
stable-diffusion.cpp's output before it replaced it (details in [docs/PERFORMANCE.md](docs/PERFORMANCE.md)):

- the Qwen-Image (Wan 2.1) VAE: for one frame the causal 3D VAE is exactly a 2D VAE (only the last temporal
  slice of each kernel sees the image), so it runs as fast as an SDXL VAE (0.27 s instead of ~45 s);
- the Anima DiT (Cosmos-Predict2 style, with its LLM adapter), 2x faster per step, LoRA / LoHa / LoKr merged in;
- the Qwen3-0.6B text encoder, the Qwen2 BPE and T5 unigram tokenizers and the `(tag:1.2)` prompt weights
  (token ids identical to stable-diffusion.cpp's);
- the sampler: Philox noise, the shift-3 discrete-flow schedules, Euler / Euler a / DPM++ 2M, CFG, img2img,
  masked img2img for face repair and the EasyCache step cache. A one-step image is bit-identical to
  stable-diffusion.cpp's; full images match visually (same seed, same picture).

The ESRGAN upscaler runs in MLX too. `.pth` files are read by a small loader that only accepts plain
tensor dictionaries and never executes pickle code.

## Credits

- [mlx-swift](https://github.com/ml-explore/mlx-swift) and the StableDiffusion code of
  [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) (MIT, Apple; see `THIRD_PARTY_LICENSE_mlx-swift-examples.txt`)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT, see `THIRD_PARTY_LICENSE_stable-diffusion.cpp.txt`): the WAI-ANIMA
  pipeline, tokenizers, sampler, ADetailer geometry and YOLOv8 decoding are ported from it; the Qwen2 BPE merges
  and T5 vocabulary in `Sources/WaiMLX/Resources` are the ones it embeds (Qwen and Google T5 tokenizers, Apache-2.0)
- [Character Select SAA](https://github.com/mirabarukaso/character_select_stand_alone_app) (MIT) for the character data and the workflow this app reproduces
- [Illustrious & NoobAI Style Explorer](https://github.com/ThetaCursed/Illustrious-NoobAI-Style-Explorer) by ThetaCursed (MIT) for the artist style list and samples
- Sampling and checkpoint-key conventions follow [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and k-diffusion; CLIP BPE vocabulary from OpenAI CLIP (MIT)
- The WAI models are by WAI0731; check each model's license on Civitai before using the images

## License

MIT
