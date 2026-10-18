import random
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi


def _blur(a, s):
    return ndi.gaussian_filter(a, sigma=s, mode="nearest")


@lru_cache(None)
def _dots(rmax):
    out = {}
    for r in range(1, rmax + 1):
        g = np.arange(-r, r + 1, dtype=np.float32)
        xx, yy = np.meshgrid(g, g, indexing="xy")
        d2 = xx * xx + yy * yy
        sigma = max(0.4, r * 0.35)
        k = np.exp(-0.5 * d2 / (sigma * sigma)) * (d2 <= r * r)
        out[r] = (k / k.max()).astype(np.float32)
    return out


def _ellipse(radius, elongation, angle):
    g = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(g, g, indexing="xy")
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr, yr = xx * cos_a + yy * sin_a, -xx * sin_a + yy * cos_a
    d2 = (xr / max(radius * elongation, 0.5)) ** 2 + (yr / max(radius, 0.5)) ** 2
    k = np.exp(-0.5 * d2 * (radius / max(0.4, radius * 0.4)) ** 2) * (d2 <= 1.0)
    return (k / k.max()).astype(np.float32)


def _sub(img, k, x, y, v):
    h, S = k.shape[0] // 2, img.shape[0]
    y1, y2, x1, x2 = max(0, y - h), min(S, y + h + 1), max(0, x - h), min(S, x + h + 1)
    if y2 > y1 and x2 > x1:
        img[y1:y2, x1:x2] -= v * k[h - (y - y1):h + (y2 - y), h - (x - x1):h + (x2 - x)]


def _bead_positions(fib, spacing, rng):
    if not fib.any():
        return []
    d = ndi.distance_transform_edt(fib)
    ys, xs = np.nonzero(d >= max(1.0, d.max() * 0.3))
    order = np.arange(len(xs))
    rng.shuffle(order)
    cell, grid, out, s2 = max(1, int(spacing)), {}, [], spacing * spacing
    for k in order:
        x, y = xs[k], ys[k]
        gx, gy = x // cell, y // cell
        if all((x - px) ** 2 + (y - py) ** 2 >= s2 for i in (-1, 0, 1) for j in (-1, 0, 1) for px, py in grid.get((gx + i, gy + j), ())):
            out.append((x, y))
            grid.setdefault((gx, gy), []).append((x, y))
    return out


def _beads(img, fib, rng, spacing, radius):
    S, img = img.shape[0], img.copy()
    for x, y in _bead_positions(fib, spacing, rng):
        jx, jy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        bx, by = int(np.clip(x + jx, 0, S - 1)), int(np.clip(y + jy, 0, S - 1))
        r = rng.integers(max(1, radius - 1), radius + 2)
        k = _ellipse(r, rng.uniform(1.0, 1.8), rng.uniform(0, 2 * np.pi))
        _sub(img, k, bx, by, rng.uniform(0.1, 0.16))
        if rng.random() < 0.3:
            a, dist = rng.uniform(0, 2 * np.pi), rng.uniform(r * 0.5, r * 1.2)
            mx, my = int(np.clip(bx + dist * np.cos(a), 0, S - 1)), int(np.clip(by + dist * np.sin(a), 0, S - 1))
            mr = rng.integers(max(1, r - 1), r + 1)
            k = _ellipse(mr, rng.uniform(1.0, 1.8), rng.uniform(0, 2 * np.pi))
            _sub(img, k, mx, my, rng.uniform(0.1, 0.16) * 0.8)
    return np.clip(img, 0.0, 1.0)


def _particles(img, rng, rmax, density=0.0015):
    S = img.shape[0]
    n = int(rng.poisson(density * float(S * S)))
    if n <= 0:
        return img
    xs, ys, radii = rng.integers(0, S, size=n), rng.integers(0, S, size=n), rng.integers(1, rmax + 1, size=n)
    vals, img = rng.uniform(0.15, 0.32, size=n), img.copy()
    for r, k in _dots(rmax).items():
        idx = radii == r
        if idx.any():
            imp = np.zeros_like(img, dtype=np.float32)
            np.add.at(imp, (ys[idx], xs[idx]), vals[idx])
            img -= ndi.convolve(imp, k, mode="nearest")
    return np.clip(img, 0.0, 1.0)


def _speckle(img, rng):
    img = img.copy()
    for p, lo, hi, sign in ((0.025, 0.08, 0.14, -1), (0.008, 0.04, 0.10, 1)):
        m = rng.random(size=img.shape) < p
        v = rng.uniform(lo, hi, size=img.shape).astype(np.float32)
        img[m] += sign * v[m]
    return np.clip(img, 0.0, 1.0)


def render(mask, seed):
    S = mask.shape[0]
    ds = 4096 // S
    rng = np.random.default_rng(random.Random(seed).randint(0, 2**32 - 1))
    fib = ndi.binary_dilation(mask, iterations=1)
    bg = np.full((S, S), 0.76, dtype=np.float32)
    bg += rng.normal(0.0, 0.055, size=bg.shape).astype(np.float32)
    bg = _blur(bg, 0.7)
    bg += _blur(rng.normal(0.0, 0.012, size=bg.shape).astype(np.float32), max(1.0, 0.22 * S))
    rng.normal(0.0, 0.0, size=bg.shape)
    img = _speckle(_particles(_beads(bg, fib, rng, 13.0 // ds, 12 // ds), rng, 16 // ds), rng)
    m, s = float(img.mean()), float(img.std())
    img = (img - m) * (0.12 / s) + 0.7 if s > 1e-5 else img - m + 0.7
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
