#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
APP_NAME="Claude Toolbar"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
DMG_PATH="$DIST_DIR/claude-toolbar.dmg"

if ! command -v hdiutil >/dev/null 2>&1; then
  echo "hdiutil not found; building a DMG requires macOS" >&2
  exit 1
fi

rm -rf "$BUILD_DIR" "$DIST_DIR"

python3 setup.py py2app

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "Expected app bundle at $APP_BUNDLE but it was not created" >&2
  exit 1
fi

hdiutil create -volname "$APP_NAME" -srcfolder "$APP_BUNDLE" -ov -format UDZO "$DMG_PATH"

echo "DMG created at $DMG_PATH"
