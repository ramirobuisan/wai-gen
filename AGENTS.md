# Agent notes for WAI Gen

Native macOS SwiftUI app; image generation runs in MLX (Sources/WaiMLX). Current task: docs/plans/noobai-vpred.md.

- Work on the branch you're on. Never push, tag, release or touch main. Commit locally with small, described commits.
- Build the CLI: `xcodebuild -scheme waigen-cli -destination 'platform=macOS,arch=arm64,variant=macos' -derivedDataPath .build-xcode -configuration Release build > build-cli.log 2>&1` (errors: `grep error: build-cli.log`)
- Build the app: `scripts/make-app.sh` (output build/WAI Gen.app). Don't install it.
- CLI binary: `.build-xcode/Build/Products/Release/waigen-cli`; models in `/Users/cookies/ComfyUI-Shared/models`.
- Every new engine option must default to the current behaviour: existing presets must give bit-identical images.
- Don't change docs/PERFORMANCE.md numbers you didn't measure. Write what you measured in the plan file's log section.
