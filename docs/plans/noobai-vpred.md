# Plan: NoobAI-XL V-Pred 1.0 in WAI Gen

Goal: a fourth preset, "NoobAI-XL V-Pred 1.0", on the existing SDXL engine (WaiSDXL), for
`checkpoints/noobaiXLNAIXL_vPred10Version.safetensors` (already downloaded; the checkpoint has `v_pred` and `ztsnr`
marker tensors). It is SDXL with **v-prediction** and a **zero-terminal-SNR** schedule. Everything else (UNet,
CLIPs, VAE, LoRAs, hi-res, face repair, compile) is unchanged.

Model card (Civitai 833294, version 1190596): Euler only ("other samplers will not work properly", no Karras),
CFG 4-5, 28-35 steps, ~1024² (832×1216), CFG Rescale recommended against oversaturation / grey images.
Prompt prefix `masterpiece, best quality, newest, absurdres, highres, safe`; negative as NoobAI EPS
(`ModelCatalog.noobaiNegative`). Diffusers: `EulerDiscreteScheduler(prediction_type="v_prediction",
rescale_betas_zero_snr=True)`.

## Steps (commit after each)

1. **Parameter.** `SDXLParameters.prediction: Prediction = .epsilon` (enum `epsilon`, `v`) and
   `cfgRescale: Float = 0` in Sources/WaiMLX/WaiSDXL.swift. Defaults keep today's behaviour.
2. **Zero-SNR schedule.** In Sources/WaiMLX/Sampling.swift `DiscreteSchedule.init` add `zeroTerminalSNR: Bool = false`.
   When true, rescale (Lin et al. 2023, as ComfyUI `ModelSamplingDiscrete(zsnr=True)`):
   `s = sqrt(alphas_cumprod)`; `s0 = s[0]`, `sT = s[999]`; `s = (s - sT) * s0 / (s0 - sT)`;
   `alphas_cumprod = s²`, then set `alphas_cumprod[999] = 4.8973451890853435e-08` (keeps sigma finite, ~4500).
   `sigmas = sqrt((1 - acp) / acp)`. WaiSDXL keeps `let schedule = DiscreteSchedule()`; make it a var chosen from
   the prediction type (or build two and pick per call).
3. **Denoiser.** In `WaiSDXL.denoiser(...)` the eps path returns `x - sigma * e`. For `.v` (k-diffusion VDenoiser,
   sigma_data = 1): `cIn = 1/sqrt(σ²+1)` (already used), `cSkip = 1/(σ²+1)`, `cOut = -σ/sqrt(σ²+1)`,
   `denoised = x * cSkip + v * cOut`, where `v` is the guided model output (same place `e` is). Do it in both
   branches of `denoiser` (with and without the `guided:` rule). Face repair (Sources/WaiMLX/FaceRepair.swift) calls
   `denoiser` too, so it follows automatically.
4. **CFG rescale** (only when `cfgRescale > 0`): with `vC`, `vU` the cond/uncond outputs and `vCFG = vU + cfg·(vC − vU)`:
   `std` over all axes but the batch; `vR = vCFG * std(vC) / std(vCFG)`; `v = cfgRescale·vR + (1 − cfgRescale)·vCFG`.
   Apply before `denoised` is formed. 0.7 is the usual value.
5. **Noise start.** Sampling starts at `x = noise * sqrt(1 + σ0²)` (eps "max denoise"); keep that for v. Check that
   σ0 ≈ 4500 gives a finite first step (Euler: `x + (x - d)/σ * (σnext - σ)`).
6. **Hi-res / refine.** `refinePass` takes the tail of `schedule.sigmas(...)`; it must use the same zero-SNR schedule.
7. **CLI.** `waigen-cli --vpred` (sets `.v`, the zero-SNR schedule) and `--cfg-rescale 0.7`; also auto-detect: if the
   checkpoint has a `v_pred` tensor, use `.v` unless told otherwise. Print which one is used.
8. **Test (CLI).** Same seed, prompt `masterpiece, best quality, newest, absurdres, highres, safe, 1girl, solo, adult,
   upper body, long hair, white shirt, denim jacket, smile, looking at viewer, city street, day`, 832×1216, Euler,
   28 steps, CFG 4.5:
   - V-Pred checkpoint with `.epsilon`: should come out broken (proves the switch matters).
   - With `.v`: a clean image. Then `--cfg-rescale 0.7`: less saturated / more natural.
   - With 1.5× hi-res (`--hires 1.5 --hires-steps 15 --hires-denoise 0.4 --hires-upscaler .../RealESRGAN_x4plus_anime_6B.pth`)
     and `--face-repair .../detectors/face_yolov8m.safetensors`.
   - Regression: WAI v16 and NoobAI-XL 1.1 (EPS) with the same seed must be bit-identical to before your change
     (save one image of each from `main` first and compare file hashes or pixels).
9. **App.** Sources/WAIGen/Core/ModelCatalog.swift: preset `noobai-xl-vpred-10`, name "NoobAI-XL V-Pred 1.0",
   engine `.sdxl`, checkpoint `checkpoints/noobaiXLNAIXL_vPred10Version.safetensors`, 832×1216, Euler (`.euler`,
   scheduler `.normal`), 28 steps, CFG 4.5, hi-res 1.5/15/0.4, `fast` = Euler 20 (no Karras). It needs a way to say
   "v-prediction + zero-SNR + CFG rescale 0.7": add fields to `ModelPreset` (e.g. `prediction`, `cfgRescale`), carry them
   on `GenerationJob` (Sources/WAIGen/Engine/GenerationService.swift) into `sdxlParameters(_:)`. Remote jobs
   (Sources/WaiRemote/RemoteProtocol.swift) only send the preset id, so the server side reads it from the preset.
10. **Log** what you measured below (times, what looked right or wrong). Don't release.

Note: the Step cache only applies with Euler a (Sampler.supportsStepCache), so it's off for this model; Faster hi-res
guidance still applies.

## Log

- Speed (M-series, 832×1216, 28 steps, Euler): v-pred runs at ~0.46-0.51 s/it, same as the
  EPS presets; auto-detection (`v_pred` marker tensor in the safetensors header) picks `.v`
  and prints it, no flag needed.
- Regression (step 8): back-to-back runs of the old (`main`, worktree at /private/tmp/opencode/wai-main)
  and new release binaries are bit-identical for WAI v16 (sha 8fbae83d…) and NoobAI-XL 1.1
  (sha e90f3e5c…), seed 42, same prompt. A stray non-matching hash from an early old-run was a
  cold-start artifact (first run after compile differs); re-running old-vs-new back-to-back matches.
- V-Pred ckpt forced with `--prediction epsilon`: output is all-black (broken, as expected —
  proves the switch matters). Default `.v`: clean image. `--cfg-rescale 0.7`: visibly softer,
  more natural colors, less oversaturation — kept as the preset value.
- Hi-res + face repair end-to-end on v-pred: `--hires 1.5 --hires-steps 15 --hires-denoise 0.4`
  with RealESRGAN_x4plus_anime_6B + `--face-repair face_yolov8m` → 35.45 s total, 1248×1824 out,
  no grey/blank artifacts; refine pass re-noise goes through the same zero-SNR schedule.
- Implementation notes: c_in = 1/√(σ²+1) is applied to the sample before the model call for
  *both* branches (ComfyUI's `x_input`); moving it was what initially broke EPS bit-exactness.
  CFG Rescale works in model-output space before the eps/v→x0 conversion, fixed 0.7 blend
  (ComfyUI convention), so `cfgRescale` acts as an on/off switch. Zero-SNR is a schedule-level
  flag (`DiscreteSchedule(zeroTerminalSNR:)`, σ_max ≈ 4519, first step finite).
- Preset wiring (step 9): `prediction`/`cfgRescale` live on `PresetDefaults` only, resolved from
  `job.preset.defaults` inside `sdxlParameters(_:)` — not added to persisted `GenerationSettings`
  and no new UI; sampler/scheduler/prefix/negative already propagate via `settings.apply(preset)`.
  No job/CLI/remote plumbing needed (jobs carry the preset, remote sends its id).
  Release build clean; app launches with the four presets. No release cut.

