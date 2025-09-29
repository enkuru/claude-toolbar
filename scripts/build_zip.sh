#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
APP_NAME="Claude Toolbar"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
ZIP_PATH="${ZIP_PATH:-$DIST_DIR/claude-toolbar.zip}"

rm -rf "$BUILD_DIR" "$DIST_DIR"

python3 setup.py py2app

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "Expected app bundle at $APP_BUNDLE but it was not created" >&2
  exit 1
fi

rm -f "$ZIP_PATH"

if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE" "$ZIP_PATH"
else
  (cd "$DIST_DIR" && zip -ry "${ZIP_PATH##*/}" "$APP_NAME.app")
fi

echo "App bundle created at $APP_BUNDLE"
echo "Zip archive created at $ZIP_PATH"
