# Agent notes for WAI Gen

Native macOS SwiftUI app; image generation runs in MLX (Sources/WaiMLX). Current task: docs/plans/z-image-anime.md.

- You work in the worktree `/Users/cookies/Developer/wai-gen-zimage` on branch `feat/z-image-anime`. Check
  `git branch --show-current` says `feat/z-image-anime` before every commit. Never switch branches, never commit on
  `main`, never push, tag, release, merge or rebase. Don't touch `/Users/cookies/Developer/wai-gen` (another branch
  is being reviewed there). Commit locally with small, described commits.
- Build the CLI: `xcodebuild -scheme waigen-cli -destination 'platform=macOS,arch=arm64,variant=macos' -derivedDataPath .build-xcode -configuration Release build > build-cli.log 2>&1` (errors: `grep error: build-cli.log`). First build takes ~10 min.
- Build the app: `scripts/make-app.sh` (output build/WAI Gen.app). Don't install it; never touch ~/Applications.
- CLI binary: `.build-xcode/Build/Products/Release/waigen-cli`; models in `/Users/cookies/ComfyUI-Shared/models`
  (read-only: never move, rename or delete model files). Test outputs and reference dumps go in
  `/Users/cookies/Developer/zimage-ref/`, not in the repo.
- Every new engine option must default to the current behaviour: existing presets must give bit-identical images.
- Don't change docs/PERFORMANCE.md numbers you didn't measure. Write what you measured in the plan file's log section.
- If a step fails three times the same way, stop, write what you tried in the log, and wait.
