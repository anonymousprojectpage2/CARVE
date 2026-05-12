#!/bin/bash
# CARVE environment helper
set -e
: "${OPENVLA_REPO:?Please set OPENVLA_REPO=/path/to/openvla}"
CARVE_DIR=$(cd "$(dirname "$0")" && pwd)
PKG_SRC="$CARVE_DIR/carve"
TARGET="$OPENVLA_REPO/carve"
if [ -e "$TARGET" ] && [ ! -L "$TARGET" ]; then
    echo "[setup] ERROR: $TARGET exists and is not a symlink."
    exit 1
fi
[ -L "$TARGET" ] && rm "$TARGET"
ln -s "$PKG_SRC" "$TARGET"
echo "[setup] linked $PKG_SRC -> $TARGET"
