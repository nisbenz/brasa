#!/usr/bin/env bash
# Download the prebuilt harmonized dataset (~280 MB, includes the U-Net ensemble data) from this repo's GitHub release,
# so the app runs without downloading 9 GB from FIRMS or rerunning the pipeline.
# Needs the GitHub CLI (gh) signed in with access to the repo.
# Usage: pipeline/fetch_prebuilt.sh [data_dir]   (default ~/firms/derived, which the app reads by default)
set -euo pipefail
DIR="${1:-$HOME/firms/derived}"
TAG="data-2000-2025"
ASSET="brasa-data-2000-2025.tar.gz"
REPO="nisbenz/brasa"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

gh release download "$TAG" --repo "$REPO" --pattern "$ASSET*" --dir "$TMP"
(cd "$TMP" && sha256sum -c "$ASSET.sha256")
mkdir -p "$DIR"
tar -xzf "$TMP/$ASSET" -C "$DIR" --strip-components=1
echo "Prebuilt data unpacked into $DIR"
[ "$DIR" = "$HOME/firms/derived" ] || echo "Start the app with: BRASA_DATA=$DIR uvicorn server:app --port 8765"
