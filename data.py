import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
DATA = Path(os.environ.get("FORKDET_DATA", "data"))
REAL_REPO, SYN_REPO = "jhehli/tem-fork-detection", "jhehli/synthetic-tem-images"
CLASSES = ("replication", "reversed")
REAL_MAP = {"Normal Fork": 0, "Reversed Fork": 1, "Crossing": 1}
SYN_MAP = {"replication_fork": 0, "reversed_fork": 1}
SCALE, FRAME = 0.5, 4096


def download():
    from huggingface_hub import snapshot_download
    get = lambda repo, d, pat: snapshot_download(repo, repo_type="dataset", local_dir=DATA / d, allow_patterns=pat, max_workers=16)
    get(REAL_REPO, "real", ["test/*", "training/annotations.csv"])
    names = pd.read_csv(DATA / "real/training/annotations.csv").image.unique()
    get(REAL_REPO, "real", [f"training/images/{n}.tif" for n in names])
    get(SYN_REPO, "synthetic", ["*/manifest.csv", "*/tem/*/*.png"])


def stretch(a, lo=1, hi=99):
    a = a.astype(np.float32)
    l, h = np.percentile(a, (lo, hi))
    return (np.clip((a - l) / max(h - l, 1e-6), 0, 1) * 255).astype(np.uint8)


def load_image(path, scale=SCALE, cache=True):
    out = DATA / "cache" / f"s{scale}" / Path(path).relative_to(DATA).with_suffix(".npy")
    if cache and out.exists():
        return np.load(out)
    img = Image.fromarray(stretch(np.array(Image.open(path))))
    size = round(FRAME * scale)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BOX)
    img = np.asarray(img)
    if cache:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, img)
    return img


def split_val(recs, frac, key, seed=0):
    rng = np.random.default_rng(seed)
    for k in sorted({key(r) for r in recs if r["split"] == "train"}):
        idx = [i for i, r in enumerate(recs) if r["split"] == "train" and key(r) == k]
        for i in rng.permutation(idx)[: int(round(frac * len(idx)))]:
            recs[i]["split"] = "val"
    return recs


def real_records(val_frac=0.15):
    recs = []
    for split, csv, d, ext in [("train", "training/annotations.csv", "training/images", ".tif"), ("test", "test/labels.csv", "test/images", ".png")]:
        df = pd.read_csv(DATA / "real" / csv)
        for name, g in df.groupby("image", sort=True):
            g = g[g.label.isin(REAL_MAP)]
            pts = np.c_[g.x, g.y, g.label.map(REAL_MAP)].astype(np.float32).reshape(-1, 3)
            sample = re.sub(r"^\d{8}_", "", re.split(r"_tile_?set", name.lower())[0])
            recs.append(dict(id=name, path=str(DATA / "real" / d / f"{name}{ext}"), split=split, points=pts, group=sample))
    return split_val(recs, val_frac, key=lambda r: tuple(c in r["points"][:, 2] for c in (0, 1)))


def synthetic_records(val_frac=0.1):
    recs = []
    for split in ["train", "test"]:
        df = pd.read_csv(DATA / "synthetic" / split / "manifest.csv").rename(columns={"class": "cls"})
        for r in df.itertuples():
            pts = np.array([[r.junction_x, r.junction_y, SYN_MAP[r.cls]]] if r.cls in SYN_MAP else [], np.float32).reshape(-1, 3)
            recs.append(dict(id=Path(r.tem).stem, path=str(DATA / "synthetic" / split / r.tem), split=split, points=pts, group=f"f{r.fiber_min}-{r.fiber_max}", cls=r.cls))
    return split_val(recs, val_frac, key=lambda r: (r["cls"], r["group"]))


def records(name, **kw):
    return {"real": real_records, "synthetic": synthetic_records}[name](**kw)


def load_split(name, split, scale=SCALE):
    recs = [r for r in records(name) if r["split"] == split]
    return recs, [load_image(r["path"], scale) for r in recs], [r["points"] * [scale, scale, 1] for r in recs]


def sample_crop(imgs, pts, size, pos_frac, rng):
    pos = [(i, k, int(p[k, 2])) for i, p in enumerate(pts) for k in range(len(p))]
    i = rng.integers(len(imgs))
    H, W = imgs[i].shape
    x0, y0 = rng.integers(0, W - size + 1), rng.integers(0, H - size + 1)
    if pos and rng.random() < pos_frac:
        cls = sorted({c for *_, c in pos})
        c = cls[rng.integers(len(cls))]
        cand = [(i, k) for i, k, cc in pos if cc == c]
        i, k = cand[rng.integers(len(cand))]
        H, W = imgs[i].shape
        m = size // 8
        cx, cy = pts[i][k, :2]
        x0 = int(np.clip(rng.integers(int(cx) - size + m, int(cx) - m + 1), 0, W - size))
        y0 = int(np.clip(rng.integers(int(cy) - size + m, int(cy) - m + 1), 0, H - size))
    p = pts[i] - [x0, y0, 0]
    p = p[(p[:, 0] >= 0) & (p[:, 0] < size) & (p[:, 1] >= 0) & (p[:, 1] < size)]
    img = imgs[i][y0:y0 + size, x0:x0 + size]
    for _ in range(rng.integers(4)):
        img, p = np.rot90(img), np.c_[p[:, 1], size - p[:, 0], p[:, 2]]
    if rng.random() < 0.5:
        img, p = img[:, ::-1], np.c_[size - p[:, 0], p[:, 1], p[:, 2]]
    return np.ascontiguousarray(img), p.astype(np.float32).reshape(-1, 3)


def jitter(x, gamma=(0.8, 1.2), noise=0.03):
    g = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(*gamma)
    return (x.clamp(1e-4, 1) ** g + noise * torch.randn_like(x)).clamp(0, 1)


def _cache(path):
    load_image(path)


if __name__ == "__main__":
    from multiprocessing import Pool
    download()
    paths = [r["path"] for n in ("real", "synthetic") for r in records(n)]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        pool.map(_cache, paths, chunksize=4)
