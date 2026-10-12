#!/usr/bin/env python3
"""The union of captured kernel sets (aot.json + cubins/): OUT SRC... ; a hash seen in an earlier set is kept once."""

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    out, *srcs = map(Path, sys.argv[1:])
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    kernels, seen = [], set()
    for src in srcs:
        for k in json.loads((src / "aot.json").read_text())["kernels"]:
            if k["hash"] in seen:
                continue
            seen.add(k["hash"])
            kernels.append(k)
            shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", out / "cubins" / f"{k['hash']}.cubin")
    kernels.sort(key=lambda x: (x["fn"], x["hash"]))
    (out / "aot.json").write_text(json.dumps({"generator": "kernel_union.py", "kernels": kernels}, indent=1) + "\n")
    print(f"{len(kernels)} kernels -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
