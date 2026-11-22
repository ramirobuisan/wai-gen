# Plan: Z-Image Turbo Anime in WAI Gen (MLX port)

Goal: a new MLX engine, **WaiZImage**, and a preset "Z-Image Anime (Turbo)" for
`/Users/cookies/ComfyUI-Shared/models/checkpoints/z-image-anime-aio-bf16.safetensors` (20.5 GB, already downloaded,
Civitai 2259646 v2543657). The file is all-in-one: DiT + text encoder + VAE. It is a distilled "Turbo" model:
**~9 steps, CFG 1 (one model call per step, no negative prompt)**, flow matching with shift 3.

Reference implementation (read it, it is the source of truth): the local ComfyUI at
`/Users/cookies/ComfyUI-Installs/ComfyUI/ComfyUI` (Python with torch: `.venv/bin/python3` in that folder).
- DiT: `comfy/ldm/lumina/model.py` (`NextDiT` with `z_image_modulation=True`; `JointAttention`, `FeedForward`,
  `JointTransformerBlock`, `FinalLayer`, `pad_zimage`, `pos_ids_x`, `patchify_and_embed`, `_forward`).
- Config: `comfy/model_detection.py` around the `dim == 3840  # Z image` branch; `comfy/supported_models.py`
  classes `Lumina2` / `ZImage`; `comfy/model_base.py` class `Lumina2` (`extra_conds`).
- Text encoder: `comfy/text_encoders/z_image.py` + `comfy/text_encoders/llama.py` (`Qwen3_4B`).
- Latent format: `comfy/latent_formats.py` class `Flux` (16 channels, scale 0.3611, shift 0.1159).
- In ComfyUI this model ran 1024², 9 steps at **18.9 s warm** (MPS). Target: faster than that in MLX.

## What the checkpoint contains (checked from its header)

| prefix | what | notes |
|---|---|---|
| `model.diffusion_model.` | NextDiT, dim 3840, **30** `layers`, **2** `noise_refiner`, **2** `context_refiner` | 6.2 B params, bf16 |
| `text_encoders.qwen3_4b.transformer.model.` | Qwen3-4B: 36 layers, hidden 2560, 32 q heads / 8 kv heads (head 128), MLP 9728, q_norm/k_norm | 4.0 B params |
| `vae.encoder.` / `vae.decoder.` | Flux VAE (SD-style AutoencoderKL, **16 latent channels, no quant_conv / post_quant_conv**) | 84 M params |

DiT config (model_detection.py): `n_heads 30`, `n_kv_heads 30` (head dim 128), `axes_dims [32, 48, 48]`,
`axes_lens [1536, 512, 512]`, `rope_theta 256`, `ffn hidden 10240`, `z_image_modulation True`, `time_scale 1000`,
`pad_tokens_multiple 32` (because `cap_pad_token` exists), patch 2×2 on 16 channels (x_embedder 64 → 3840).
Key shapes: `layers.N.attention.qkv [11520, 3840]` (fused q,k,v), `attention.out [3840, 3840]`,
`q_norm/k_norm [128]` (RMSNorm per head), `attention_norm1/2`, `ffn_norm1/2 [3840]`,
`feed_forward.w1/w3 [10240, 3840]`, `w2 [3840, 10240]` (SwiGLU: w2(silu(w1 x) * w3 x)),
`adaLN_modulation.1 [15360, 256]` (4 × 3840: scale_msa, gate_msa, scale_mlp, gate_mlp — check order in the code),
`t_embedder.mlp.0 [1024, 256]`, `.mlp.2 [256, 1024]`, `cap_embedder.0 [2560]` (RMSNorm) + `.1 [3840, 2560]`,
`final_layer.adaLN_modulation.1 [3840, 256]`, `final_layer.linear [64, 3840]`, `norm_final [3840]`,
`x_pad_token`, `cap_pad_token [1, 3840]`. `context_refiner` blocks have **no** adaLN.

Facts from ComfyUI's `_forward` you must match:
- `t = 1.0 - timesteps` (timesteps = sigma, model multiplier 1.0), embedded as `t_embedder(t * 1000)`.
- Sequence order is **caption tokens first, then image tokens**; caption and image are padded to a multiple of 32 with
  `cap_pad_token` / `x_pad_token`. Image position ids: `pos_ids_x` (axis 0 = a start index after the caption,
  axis 1 = row, axis 2 = column). Caption position ids: read `patchify_and_embed`.
- The returned prediction is **negated** (`return -img`). With that, the sampler sees a flow-matching velocity and
  `denoised = x - sigma * out` (ComfyUI `ModelType.FLOW`, CONST: `x0 = x - σ·v`).
- Text conditioning = Qwen3-4B **hidden state of layer index 34** (`layer_idx=-2`: output of the second-to-last
  layer), **no final norm**, prompt wrapped as `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
  (ComfyUI default `thinking=True` adds nothing else). Tokenizer: Qwen2 BPE, same vocabulary as WAI-ANIMA's
  (`Qwen2Tokenizer` in Sources/WaiMLX/AnimaText.swift, `Resources/qwen2_merges.txt`); the special tokens
  `<|im_start|>` = 151644, `<|im_end|>` = 151645 must be emitted as single ids.

## What already exists in WAI Gen (reuse it)

- `Sources/WaiMLX/Qwen3TextEncoder.swift`: generic Qwen3 (reads sizes from the weights). Needs: a key **prefix**
  parameter (`text_encoders.qwen3_4b.transformer.model.`) that keeps only those keys (don't materialise the DiT/VAE
  tensors twice), and an option to **stop after layer `layers - 2` without the final norm**.
- `Sources/WaiMLX/WaiAnima.swift`: flow matching with shift 3 (`time_snr_shift`), Euler, flow Euler ancestral
  (`get_ancestral_step_flow`), DPM++ 2M, discrete / Karras / simple schedules, Philox noise, progress/cancel plumbing,
  face repair hook. Model the new pipeline on it.
- `Sources/WaiMLX/VAE.swift` + `Configuration.swift`: the SD AutoencoderKL. Flux VAE is the same architecture with
  `latent_channels 16` and **without** `quant_conv`/`post_quant_conv` → add a flag, and the Flux latent scaling:
  `latent = (z - 0.1159) * 0.3611` on encode, `z = latent / 0.3611 + 0.1159` before decode.
- `Sources/WaiMLX/CheckpointLoader.swift`: safetensors loading + `safetensorsTensorNames` pattern.
- `Sources/WaiMLX/AnimaDiT.swift`: an MLX DiT with RoPE, RMSNorm, adaLN, `MLXFast.scaledDotProductAttention`
  — copy its style (and its `compile` usage) for NextDiT.

## Steps (commit after each, on branch feat/z-image-anime only)

0. **Reference tensors.** Write `scripts/zimage_reference.py`, run with ComfyUI's venv python from the ComfyUI folder
   (add the folder to `sys.path` and set `sys.argv = [sys.argv[0]]` before importing comfy, which parses argv; MPS is
   fine, use `[sys.argv[0], "--cpu"]` only if MPS fails; save tensors as float32).
   Load the AIO with `comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=True)`.
   For the fixed prompt `anime, 1girl, solo, long hair, white shirt, denim jacket, smile, city street, day` save to
   `/Users/cookies/Developer/zimage-ref/ref.safetensors` (outside the repo):
   token ids; text conditioning (`clip.encode_from_tokens(tokens, return_pooled=False)`), a seeded latent noise
   `[1, 16, 128, 128]` (1024²); the DiT output for that latent at sigma 1.0 and 0.5 (call
   `model.model.diffusion_model(x, timesteps, context, num_tokens)` directly, and also note what `model.model.apply_model`
   returns); `vae.decode` of the noise latent's first 64×64 crop. Log shapes, dtypes and min/max/mean of each.
1. **Text encoder.** Prefix + layer -2 options on `Qwen3TextEncoder`; chat template + special tokens. CLI check
   (`waigen-cli --zimage-check ref.safetensors`) prints token ids equality and max abs diff of the conditioning.
   Target: identical ids, max abs diff small for bf16 (relative < 1e-2).
2. **DiT** in `Sources/WaiMLX/ZImageDiT.swift`: embedders, 3D RoPE (axes 32/48/48, theta 256, complex-pair rotation as
   in the reference), noise_refiner (with modulation), context_refiner (without), 30 main layers, final layer,
   padding, unpatchify, negation. Check both reference sigmas: relative error < 1e-2.
3. **VAE**: Flux VAE decode (and encode for later img2img) via the flagged Autoencoder; check against the reference decode.
4. **Pipeline** `Sources/WaiMLX/WaiZImage.swift`: `ZImageParameters` (prompt, width, height, steps 9, cfg 1, sampler
   euler / euler_a, scheduler simple, seed, shift 3). **CFG 1 = only the cond pass** (no uncond batch). If cfg > 1, the
   negative is the empty prompt encoded the same way (ComfyUI uses ConditioningZeroOut on the negative: zeros; do that).
   Load the AIO file once and split by prefix.
5. **CLI**: `waigen-cli --zimage <aio.safetensors> -p "..." -W 832 -H 1216 --steps 9 --cfg 1 --sampler euler_ancestral
   --seed 42 -o out.png`, printing encode / sample / decode times like the other engines. Test at 1024² and 832×1216.
   Images should look clean, anime style, not noisy / not grey. Time it (`--repeat 2`, second run is warm).
6. **Speed**: `compile` the DiT step (as WaiSDXL/WaiAnima do), keep weights bf16, check memory (`MLX.GPU` peak).
   Optional: beta scheduler (ComfyUI `beta_scheduler`, alpha = beta = 0.6) if euler_a + simple looks worse than
   ComfyUI's euler_a + beta. Write before/after numbers in the log.
7. **App**: `EngineKind.zimage`, preset `z-image-anime-turbo` ("Z-Image Anime (Turbo)") in
   Sources/WAIGen/Core/ModelCatalog.swift: checkpoint `checkpoints/z-image-anime-aio-bf16.safetensors`, 832×1216,
   9 steps, CFG 1, Euler a, scheduler simple, no hi-res, no negative needed, no LoRAs for now. Wire it in
   Sources/WAIGen/Engine/GenerationService.swift like the Anima engine. Unsupported controls (LoRAs, hi-res, face
   repair, step cache) are disabled for this preset. `scripts/make-app.sh` must build; launch only the built copy if you
   need to look (never install it, never touch ~/Applications).
8. **Regression**: WAI v16, WAI-ANIMA and NoobAI-XL 1.1 images (same seed, CLI) must be bit-identical to a `main` build.
9. **Log** what you measured below. Don't release.

## Log
