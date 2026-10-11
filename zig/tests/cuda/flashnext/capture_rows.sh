#!/usr/bin/env bash
# The batched prompt pass's row-count kernels (capture_rows.py) added to a captured Flash Next kernel set.
# Usage: capture_rows.sh WORK BASE_KERNELS OUT_KERNELS [IMAGE]
#   WORK          a host dir this script owns (WORK/cache, WORK/out)
#   BASE_KERNELS  capture.sh's kernel set (aot.json + cubins)
#   OUT_KERNELS   the union written here (the base set's kernels plus the M > 1 forms)
# Needs only a little GPU memory: no checkpoint loads.
set -euo pipefail
WORK=$(readlink -f "$1"); BASE=$(readlink -f "$2"); OUT=$3; IMAGE=${4:-tensorfold-qwen38:v0.6.1}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
mkdir -p "$WORK/cache" "$WORK/out" "$OUT"
OUT=$(readlink -f "$OUT")
docker run --rm --name fn-capture-rows --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TENSORFOLD_NO_UPDATE_CHECK=1 -v "$WORK/cache":/cache -v "$REPO":/src:ro -v "$WORK/out":/out -w /out \
  "$IMAGE" python -u /src/zig/tests/cuda/flashnext/capture_rows.py --out /out --tools /src/tools/zig
python3 -B "$REPO/tools/zig/triton_aot_manifest.py" --cache "$WORK/cache/triton" \
  --launches "$WORK/out/launches.json" --mount /cache/triton --out "$WORK/out/manifest.json"
python3 -B "$REPO/zig/tests/cuda/nemotron/aot_pack.py" --manifest "$WORK/out/manifest.json" \
  --cache "$WORK/cache/triton" --jit "$WORK/out/jit.json" --out "$WORK/rows"
python3 -B - "$BASE" "$WORK/rows" "$OUT" <<'PY'
import json, shutil, sys
from pathlib import Path
base, rows, out = map(Path, sys.argv[1:])
(out / "cubins").mkdir(parents=True, exist_ok=True)
kernels, seen = [], set()
for src in (base, rows):
    for k in json.loads((src / "aot.json").read_text())["kernels"]:
        if k["hash"] in seen:
            continue
        seen.add(k["hash"])
        kernels.append(k)
        shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", out / "cubins" / f"{k['hash']}.cubin")
kernels.sort(key=lambda x: (x["fn"], x["hash"]))
(out / "aot.json").write_text(json.dumps({"generator": "capture_rows.sh", "kernels": kernels}, indent=1) + "\n")
print(f"{len(kernels)} kernels -> {out}")
PY
