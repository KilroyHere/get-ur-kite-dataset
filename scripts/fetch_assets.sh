#!/usr/bin/env bash
# Fetch third-party assets at pinned commits into third_party/ (git-ignored).
#
#   get_zero          LEAP Hand URDF + meshes used as the base model, and GET-Zero's
#                     generated hands (kept as an extra test set). Its repository has
#                     no license, so these files are fetched, never committed here.
#   mujoco_menagerie  LEAP Hand MJCF (MIT): hand-tuned collision boxes and tip meshes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"

GET_ZERO_SHA=279652da160823521e881952f67870b0d98674ce
MENAGERIE_SHA=b0655057a9df2abe39ea7fc989914ba6f248301c

fetch() {  # fetch <url> <sha> <dir> <sparse patterns...>
  local url=$1 sha=$2 dir=$3
  shift 3
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" = "$sha" ]; then
    echo "up to date: ${dir#"$ROOT"/} @ ${sha:0:7}"
    return
  fi
  if [ ! -d "$dir/.git" ]; then
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" sparse-checkout set --no-cone "$@"
  git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  echo "fetched: ${dir#"$ROOT"/} @ ${sha:0:7}"
}

fetch https://github.com/real-stanford/get_zero "$GET_ZERO_SHA" "$TP/get_zero" \
  '/get_zero/rl/assets/leap/leap_hand/original_refactor.urdf' \
  '/get_zero/rl/assets/leap/leap_hand/*.stl' \
  '/get_zero/rl/assets/leap/leap_hand/custom_models/*.stl' \
  '/get_zero/rl/assets/leap/leap_hand/generated/urdf/*'

fetch https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE_SHA" "$TP/mujoco_menagerie" \
  '/leap_hand/right_hand.xml' \
  '/leap_hand/assets/*' \
  '/leap_hand/LICENSE'
