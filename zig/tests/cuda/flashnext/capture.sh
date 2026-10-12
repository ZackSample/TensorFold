#!/usr/bin/env bash
# Flash Next (qwen4_exp) capture -> manifest -> kernel set for zig's flashnext-prompt and the native lanes engine.
# Usage: capture.sh WORK SNAPSHOT_DIR [IMAGE]
#   WORK          a host dir this script owns: WORK/cache (Triton + torch extension caches), WORK/out/cap, WORK/kernels
#   SNAPSHOT_DIR  the checkpoint's HF snapshot dir (its blobs must sit at ../../blobs, as in an HF cache)
#   IMAGE         TensorFold 0.6.1 image, default tensorfold-qwen38:v0.6.1
# The Python engine runs drafts off, greedy, --context 1535 (1536 cache slots: _chunks/_merge NCH=3, QSA=0, the dense
# attention the port launches), --ple-on-ssd, bf16 KV. Needs the GPU free (~78 GiB).
# CONTEXT, ONLY, TEACHERS and OUT override --context, --only, --teachers and WORK/out/cap: CONTEXT=16383
# ONLY=long3k,long12k TEACHERS=teacher32 OUT=sparse runs the prompts past the indexer's budget on a 16,384-slot cache
# (_chunks/_merge NCH=5, QSA=1, the pooled keys, scores and selection the port's sparse rows run).
set -euo pipefail
WORK=$(readlink -f "$1"); SNAP=$(readlink -f "$2"); IMAGE=${3:-tensorfold-qwen38:v0.6.1}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
HF=$(dirname "$(dirname "$(dirname "$SNAP")")")          # .../hub/models--X/snapshots/REV -> .../hub (holds the blobs)
REL=${SNAP#"$HF"/}
CONTEXT=${CONTEXT:-1535}; ONLY=${ONLY:-}; TEACHERS=${TEACHERS-teacher,teacher32}; OUT=${OUT:-cap}
mkdir -p "$WORK/cache" "$WORK/out/$OUT"
KSET=$WORK/kernels; [ "$OUT" = cap ] || KSET=$WORK/kernels-$OUT
docker run --rm --name fn-capture --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 \
  -e TENSORFOLD_MEMORY_RESERVE_GIB=2 \
  -v "$HF":/hf:ro -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out -w /out \
  "$IMAGE" python -u /src/zig/tests/cuda/flashnext/capture.py --model "/hf/$REL" --out "/out/$OUT" \
  --tools /src/tools/zig --tokens 32 --context "$CONTEXT" --only "$ONLY" --teachers "$TEACHERS"
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/$OUT/launches.json" --mount /cache/triton --out "$WORK/out/$OUT/manifest.json"
python3 -B "$REPO/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/$OUT/manifest.json" \
  --cache "$WORK/cache/triton" --jit "$WORK/out/$OUT/jit.json" --out "$KSET"
echo "kernel set: $KSET; prompt ids and references: $WORK/out/$OUT/*.ids, results.json"
