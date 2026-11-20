#!/bin/zsh
# Publish the version in VERSION as a GitHub release: the app's update channel downloads it from there.
# Usage: scripts/release.sh [--beta]   (--beta: a pre-release, only for the Beta channel)
#
# Before: bump VERSION, add a "## <version>" section at the top of CHANGELOG.md (its text becomes the release
# notes), commit and push. The script builds, zips build/WAI-Gen.zip, tags v<version> and creates the release.
set -euo pipefail

ROOT=${0:A:h:h}
cd "$ROOT"
VERSION=$(cat VERSION)
PRE=()
[[ "${1:-}" == "--beta" ]] && PRE=(--prerelease)

[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Commit your changes first"; exit 1; }
git fetch -q origin
[[ "$(git rev-parse HEAD)" == "$(git rev-parse '@{u}')" ]] || { echo "Push first (HEAD isn't origin's)"; exit 1; }
git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null && { echo "v$VERSION already exists: bump VERSION"; exit 1; }
head -20 CHANGELOG.md | grep -q "^## $VERSION" || { echo "CHANGELOG.md needs a '## $VERSION' section at the top"; exit 1; }

scripts/make-app.sh
rm -f build/WAI-Gen.zip
ditto -c -k --keepParent "build/WAI Gen.app" build/WAI-Gen.zip

NOTES=$(awk '/^## /{n++} n==1' CHANGELOG.md | tail -n +2)
git tag "v$VERSION"
git push -q origin "v$VERSION"
gh release create "v$VERSION" build/WAI-Gen.zip --title "WAI Gen $VERSION" --notes "$NOTES" --verify-tag $PRE
echo "Released v$VERSION ($(du -h build/WAI-Gen.zip | cut -f1))"
