#!/bin/zsh
# Build WAI Gen in Release and package it as build/WAI Gen.app (ad-hoc signed).
# Usage: scripts/make-app.sh [--install]   (--install copies it to ~/Applications)
set -euo pipefail

ROOT=${0:A:h:h}
PRODUCTS="$ROOT/.build-xcode/Build/Products/Release"
APP="$ROOT/build/WAI Gen.app"
VERSION=${WAIGEN_VERSION:-$(cat "$ROOT/VERSION")}

cd "$ROOT"
echo "Building (Release)…"
xcodebuild -scheme WAIGen -destination 'platform=macOS,arch=arm64,variant=macos' \
    -derivedDataPath .build-xcode -configuration Release build > build-app.log 2>&1 \
    || { grep -E "error:" build-app.log | sort -u | head -20; echo "Build failed, see build-app.log"; exit 1; }

echo "Packaging $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PRODUCTS/WAIGen" "$APP/Contents/MacOS/WAIGen"
# SwiftPM resources: WaiMLX's configs/tokenizer and MLX's Metal shader library
cp -R "$PRODUCTS/wai-gen_WaiMLX.bundle" "$APP/Contents/Resources/"
cp -R "$PRODUCTS/mlx-swift_Cmlx.bundle" "$APP/Contents/Resources/"
[ -f "$ROOT/Resources/AppIcon.icns" ] && cp "$ROOT/Resources/AppIcon.icns" "$APP/Contents/Resources/"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>WAI Gen</string>
    <key>CFBundleDisplayName</key><string>WAI Gen</string>
    <key>CFBundleIdentifier</key><string>local.waigen.app</string>
    <key>CFBundleExecutable</key><string>WAIGen</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
    <key>NSLocalNetworkUsageDescription</key><string>WAI Gen finds other Macs running it on your network to generate images there, and lets them generate on this Mac when you turn that on.</string>
    <key>NSBonjourServices</key><array><string>_waigen._tcp</string></array>
</dict>
</plist>
PLIST

codesign --force --deep --sign - "$APP" >/dev/null 2>&1
echo "Built $APP ($(du -sh "$APP" | cut -f1))"

if [[ "${1:-}" == "--install" ]]; then
    mkdir -p "$HOME/Applications"
    rm -rf "$HOME/Applications/WAI Gen.app"
    cp -R "$APP" "$HOME/Applications/"
    echo "Installed to ~/Applications/WAI Gen.app"
fi
