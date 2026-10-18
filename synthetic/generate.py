import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from synthetic.masks import CLASSES, Cfg, make_mask
from synthetic.render import render

BINS = [(1, 3), (3, 8), (8, 13), (13, 18), (18, 23), (23, 28), (28, 33), (33, 38)]


def tasks(split, count, size, bins=BINS):
    off = 1_000_000 if split == "test" else 0
    return [(split, cls, b, i, 42 + i + 100_000 * c + 10_000 * BINS.index(b) + off, size) for c, cls in enumerate(CLASSES) for b in bins for i in range(count)]


def make(task, out=None):
    split, cls, (lo, hi), i, seed, size = task
    mask, j = make_mask(cls, seed, Cfg(size, (lo, hi)))
    tem = render(mask, seed)
    name = f"{cls}/{cls}_f{lo}-{hi}_{i:04d}.png"
    if out is not None:
        for d, img in (("masks", np.stack([mask * np.uint8(255), *[np.zeros_like(mask, np.uint8)] * 2], -1)), ("tem", tem)):
            (out / split / d / cls).mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(out / split / d / name)
    row = {"split": split, "class": cls, "fiber_min": lo, "fiber_max": hi, "seed": seed, "junction_x": j[0] if j else np.nan, "junction_y": j[1] if j else np.nan, "mask": f"masks/{name}", "tem": f"tem/{name}"}
    return row, mask, tem


def _run(args):
    return make(*args)[0]


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", type=Path, default=Path(os.environ.get("FORKDET_DATA", ".")) / "synthetic_regen")
    pa.add_argument("--split", default="train", choices=["train", "test"])
    pa.add_argument("--count", type=int, default=None)
    pa.add_argument("--size", type=int, default=4096)
    pa.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    pa.add_argument("--shards", type=int, default=1)
    pa.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    pa.add_argument("--merge", action="store_true")
    a = pa.parse_args()
    d = a.out / a.split
    if a.merge:
        parts = sorted(d.glob("manifest_*.csv"))
        pd.concat(map(pd.read_csv, parts)).sort_values("seed").to_csv(d / "manifest.csv", index=False, float_format="%.2f")
    else:
        todo = tasks(a.split, a.count or (60 if a.split == "train" else 6), a.size)[a.shard::a.shards]
        with Pool(a.workers) as pool:
            rows = pool.map(_run, [(t, a.out) for t in todo], chunksize=1)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / (f"manifest_{a.shard}.csv" if a.shards > 1 else "manifest.csv"), index=False, float_format="%.2f")
