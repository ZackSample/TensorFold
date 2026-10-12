#!/usr/bin/env bash
# Flash Next's sparse attention kernels (capture_sparse.py: _pool, _scores, _chunks/_merge at QSA 1, NCH 5, bf16 and
# int8 KV with int8's _attn_prep) added to
# captured kernel sets, plus the selection fixtures flashnext-select checks fn_qsa_select against.
# Usage: capture_sparse.sh WORK OUT_KERNELS BASE_KERNELS... [-- IMAGE]
#   WORK          a host dir this script owns (WORK/cache, WORK/out, WORK/fixtures)
#   OUT_KERNELS   the union written here (every base set's kernels plus the sparse forms)
#   BASE_KERNELS  capture.sh / capture_rows.sh kernel sets (aot.json + cubins)
# Needs only a little GPU memory: no checkpoint loads.
set -euo pipefail
WORK=$(readlink -f "$1"); OUT=$2; shift 2
BASES=(); IMAGE=tensorfold-qwen38:v0.6.1
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then IMAGE=$2; break; fi
  BASES+=("$(readlink -f "$1")"); shift
done
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$WORK/cache" "$WORK/out" "$WORK/fixtures" "$OUT"
OUT=$(readlink -f "$OUT")
docker run --rm --name fn-capture-sparse --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out \
  -v "$WORK/fixtures":/fixtures -w /out "$IMAGE" \
  python -u /src/zig/tests/cuda/flashnext/capture_sparse.py --out /out --tools /src/tools/zig --fixtures /fixtures
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/launches.json" --mount /cache/triton --out "$WORK/out/manifest.json"
python3 -B "$REPO/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" \
  --cache "$WORK/cache/triton" --jit "$WORK/out/jit.json" --out "$WORK/sparse"
python3 -B "$(dirname "$0")/kernel_union.py" "$OUT" "${BASES[@]}" "$WORK/sparse"
echo "selection fixtures: $WORK/fixtures"
