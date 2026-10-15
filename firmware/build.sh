#!/usr/bin/env bash
# Assemble a per-arm AudioMoth tree from pristine upstream and build it.
#
#   ./firmware/build.sh stock    # vendor only — must reproduce the official 1.12.1 release
#   ./firmware/build.sh mfcc-dev # vendor + firmware/common + firmware/arms/mfcc-dev
#
# AM_Software/AudioMoth-<arm>/ is DISPOSABLE BUILD OUTPUT. Every run deletes and rebuilds
# it, so never edit a tree in place — the edit will vanish without warning. The tracked
# source of truth is firmware/; AM_Software/ is gitignored and holds upstream clones plus
# these generated trees.
#
# What is per-arm and what is shared:
#   arms/<arm>/   front-end, and the generated model_params.h (weights, theta,
#                 global_scale) — everything that actually differs between arms is DATA
#                 and lives here. A non-linear arm also brings its own scorer.
#   common/       the state machine and the adaptive threshold. Shared because they are
#                 parameterised by that header rather than specialised to an arm, and
#                 because each copy of the state machine is another thing that has to be
#                 kept bit-identical to birdgate/gate.py — see gate.py's opening note:
#                 if the two drift, the simulator lies.
set -euo pipefail

ARM="${1:-mfcc-dev}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AM="$HERE/../AM_Software"
VENDOR="$AM/AudioMoth-Project"
APP="$AM/AudioMoth-Firmware-Basic"
TREE="$AM/AudioMoth-$ARM"
RELEASE="$AM/AudioMoth-Firmware-Basic-1.12.1.bin"
TOOLCHAIN_PATH="${TOOLCHAIN_PATH:-$HOME/Library/Toolchains}"
TOOLCHAIN_VERSION="${TOOLCHAIN_VERSION:-gcc-arm-none-eabi-10.3-2021.10}"

[ -d "$VENDOR" ] || { echo "missing $VENDOR — see firmware/README.md" >&2; exit 1; }
[ -d "$APP" ]    || { echo "missing $APP — see firmware/README.md" >&2; exit 1; }
if [ "$ARM" != "stock" ] && [ ! -d "$HERE/arms/$ARM" ]; then
  echo "unknown arm '$ARM' — expected directory firmware/arms/$ARM" >&2; exit 1
fi

echo "==> assembling $TREE"
rm -rf "$TREE"
rsync -a --exclude '.git' --exclude 'objects' "$VENDOR/" "$TREE/"

# Upstream app layer: Firmware-Basic's sources over the Project skeleton.
cp "$APP"/src/*.c "$TREE/src/"
cp "$APP"/inc/*.h "$TREE/inc/"

# Makefile: point at the toolchain, and add the /gps/ folder Firmware-Basic needs.
sed -i '' \
  -e "s|^TOOLCHAIN_PATH = .*|TOOLCHAIN_PATH = $TOOLCHAIN_PATH|" \
  -e "s|^TOOLCHAIN_VERSION = .*|TOOLCHAIN_VERSION = $TOOLCHAIN_VERSION|" \
  -e "s|^INC = \(.*\) \.\./inc$|INC = \1 ../gps/inc ../inc|" \
  -e "s|^SRC = \(.*\) \.\./src$|SRC = \1 ../gps/src ../src|" \
  "$TREE/build/Makefile"

# This sed is the step that silently half-works if upstream reformats the Makefile, and
# the symptom would be undefined references a long way from the cause.
grep -q 'gps/inc' "$TREE/build/Makefile" && grep -q 'gps/src' "$TREE/build/Makefile" || {
  echo "Makefile patch did not apply — upstream INC/SRC lines have changed" >&2; exit 1; }

if [ "$ARM" = "stock" ]; then
  # Upstream master is 1.12.0; the released binary is 1.12.1 and the ONLY difference
  # between them is this version triple. Match it so the comparison below is exact.
  sed -i '' 's|= {1, 12, 0};|= {1, 12, 1};|' "$TREE/src/main.c"
else
  cp "$HERE"/common/*.c "$TREE/src/" 2>/dev/null || true
  cp "$HERE"/common/*.h "$TREE/inc/" 2>/dev/null || true
  cp "$HERE/arms/$ARM"/*.c "$TREE/src/" 2>/dev/null || true
  cp "$HERE/arms/$ARM"/*.h "$TREE/inc/" 2>/dev/null || true
  [ -f "$HERE/app/main.c" ] && cp "$HERE/app/main.c" "$TREE/src/main.c"
fi

echo "==> building"
make -C "$TREE/build" -s

mkdir -p "$HERE/out"
OUT="$HERE/out/birdgate-$ARM.bin"
cp "$TREE/build/audiomoth.bin" "$OUT"
echo "==> $OUT ($(stat -f%z "$OUT") bytes)"
shasum -a 256 "$OUT"

if [ "$ARM" = "stock" ] && [ -f "$RELEASE" ]; then
  # The regression check for the assembly itself. If `stock` stops reproducing the
  # official release byte for byte, the build chain has drifted and every arm built
  # from it is suspect. Run it whenever the vendor clones or this script change.
  if cmp -s "$RELEASE" "$OUT"; then
    echo "==> stock reproduces the official 1.12.1 release byte-for-byte"
  else
    echo "!! stock NO LONGER matches the official release — assembly has drifted" >&2
    cmp -l "$RELEASE" "$OUT" | head >&2
    exit 1
  fi
fi
