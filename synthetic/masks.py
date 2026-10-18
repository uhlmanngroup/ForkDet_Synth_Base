import math
import random

import numpy as np

CLASSES = ("replication_fork", "reversed_fork", "negative")


class Cfg:
    def __init__(self, size=4096, fibers=(20, 35), p_loop=0.6):
        ds = 4096 // size
        self.size, self.fibers, self.p_loop = size, fibers, p_loop
        self.thick, self.margin, self.waviness = 12 // ds, 72 // ds, 0.22
        self.hairpin_p, self.hairpin_r = 0.8, (10.0 // ds, 60.0 // ds)
        self.spacing = max(3.0, 15.0 * (size / 2048.0))
        self.angle_sep, self.attempts = math.radians(14.0), 80


def _clamp(v, c):
    return min(max(v, c.margin), c.size - c.margin)


def _smooth(pts, iterations):
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new = [pts[0]]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            new.extend([(0.75 * p1[0] + 0.25 * p2[0], 0.75 * p1[1] + 0.25 * p2[1]), (0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1])])
        new.append(pts[-1])
        pts = new
    return pts


def _wavy(rng, start, end, waviness, c, steps=6):
    scale = c.size / 2048.0
    pts = [start]
    direction = math.atan2(end[1] - start[1], end[0] - start[0])
    step_len = math.hypot(end[0] - start[0], end[1] - start[1]) / (steps + 1)
    for i in range(steps):
        base = step_len * (i + 1)
        local_dir = direction + rng.gauss(0.0, waviness * 0.5)
        lateral = local_dir + math.pi / 2
        offset = rng.gauss(0.0, waviness * 25.0 * scale)
        x = start[0] + math.cos(local_dir) * base + math.cos(lateral) * offset
        y = start[1] + math.sin(local_dir) * base + math.sin(lateral) * offset
        pts.append((_clamp(x, c), _clamp(y, c)))
    pts = _smooth(pts + [end], 2)
    dense = []
    for p1, p2 in zip(pts[:-1], pts[1:]):
        n = int(max(2, math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 6))
        dense.extend((p1[0] + (p2[0] - p1[0]) * (s / n), p1[1] + (p2[1] - p1[1]) * (s / n)) for s in range(n))
    dense.append(pts[-1])
    if len(dense) <= 2:
        return dense
    wiggle, out = waviness * 6.0 * scale, [dense[0]]
    for i in range(1, len(dense) - 1):
        (px, py), (ax, ay), (bx, by) = dense[i], dense[i - 1], dense[i + 1]
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        if seg > 0.1:
            d = rng.gauss(0.0, wiggle)
            px, py = px + -dy / seg * d, py + dx / seg * d
        out.append((px, py))
    return out + [dense[-1]]


def _hairpin(rng, start, direction, radius, c):
    radius = radius * (c.size / 1024.0)
    turn = math.pi + rng.gauss(0.0, 0.15)
    sign = rng.choice([-1, 1])
    ca = direction + sign * (math.pi / 2)
    cx, cy = start[0] + math.cos(ca) * radius, start[1] + math.sin(ca) * radius
    n, a0, arc = max(8, int(radius * 0.3)), direction - sign * (math.pi / 2), []
    for i in range(n + 1):
        a = a0 + sign * turn * (i / n)
        r = radius * (1.0 + rng.gauss(0.0, c.waviness * 0.1))
        arc.append((_clamp(cx + math.cos(a) * r, c), _clamp(cy + math.sin(a) * r, c)))
    return _smooth(arc, 1), direction + sign * turn


def _fork(rng, cls, c):
    S, m = c.size, max(c.margin, c.size // 8)
    j = (rng.uniform(m, S - m), rng.uniform(m, S - m))
    theta = rng.uniform(-math.pi, math.pi)
    if cls == "replication_fork":
        spread, asym = rng.uniform(1.05, 1.75), rng.uniform(0.9, 1.1)
        dirs, wav = [theta, theta + math.pi - (spread / 2) * asym, theta + math.pi + (spread / 2) / asym], (0.85, 1.2)
    else:
        jit = rng.uniform(-0.15, 0.15)
        dirs, wav = [theta, theta + math.pi / 2 + jit, theta + math.pi, theta + math.pi + math.pi / 2 - jit], (0.85, 1.15)
    j = (float(round(j[0])), float(round(j[1])))
    long_arm, arms = rng.randint(0, len(dirs) - 1), []
    for k, d in enumerate(dirs):
        L = rng.uniform(0.15 * S, 0.35 * S) if k == long_arm else rng.uniform(0.08 * S, 0.20 * S)
        end = (_clamp(j[0] + math.cos(d) * L, c), _clamp(j[1] + math.sin(d) * L, c))
        arm = _wavy(rng, j, end, c.waviness * rng.uniform(*wav), c)
        arm[0] = j
        arms.append(arm)
    return arms, j


def _lobe(rng, center, ra, rb, start, extent, tilt, c):
    n, (ct, st), pts = max(16, int(max(ra, rb) * 0.4)), (math.cos(tilt), math.sin(tilt)), []
    for i in range(n + 1):
        a, r = start + extent * (i / n), 1.0 + rng.gauss(0.0, c.waviness * 0.12)
        ex, ey = ra * math.cos(a) * r, rb * math.sin(a) * r
        pts.append((_clamp(center[0] + ex * ct - ey * st, c), _clamp(center[1] + ex * st + ey * ct, c)))
    return _smooth(pts, 1)


def _loop_geometry(rng, c, radii):
    m = min(max(c.margin, int(max(radii)) + c.margin), c.size // 3)
    return float(round(rng.uniform(m, c.size - m))), float(round(rng.uniform(m, c.size - m)))


def _p_shape(rng, c):
    total = rng.uniform(0.40 * c.size, 0.95 * c.size)
    frac = rng.uniform(0.40, 0.60)
    stem, ra = total * frac, total * (1.0 - frac) / (2.0 * math.pi * 0.88)
    rb = ra * rng.uniform(0.5, 1.3)
    x = _loop_geometry(rng, c, (ra, rb))
    sd, pos = rng.uniform(-math.pi, math.pi), rng.uniform(0.30, 0.70)
    above, below = stem * pos, stem * (1.0 - pos)
    top = (_clamp(x[0] + math.cos(sd) * above, c), _clamp(x[1] + math.sin(sd) * above, c))
    bot = (_clamp(x[0] + math.cos(sd + math.pi) * below, c), _clamp(x[1] + math.sin(sd + math.pi) * below, c))
    perp = sd + rng.choice([-1, 1]) * math.pi / 2
    ext = math.radians(rng.uniform(300.0, 340.0))
    start, tilt = math.pi + (2.0 * math.pi - ext) / 2.0, perp + rng.gauss(0.0, 0.3)
    a = _wavy(rng, bot, x, c.waviness * rng.uniform(0.85, 1.15), c)
    b = _lobe(rng, (x[0] + math.cos(perp) * ra, x[1] + math.sin(perp) * ra), ra, rb, start, ext, tilt, c)
    d = _wavy(rng, x, top, c.waviness * rng.uniform(0.85, 1.15), c)
    return a + b[1:] + d[1:], x


def _eight(rng, c):
    total = rng.uniform(0.40 * c.size, 0.95 * c.size)
    split = rng.uniform(0.35, 0.65)
    r1a = total * split / (2.0 * math.pi * 0.88)
    r1b = r1a * rng.uniform(0.5, 1.3)
    r2a = total * (1.0 - split) / (2.0 * math.pi * 0.88)
    r2b = r2a * rng.uniform(0.5, 1.3)
    x = _loop_geometry(rng, c, (r1a, r1b, r2a, r2b))
    spine, side = rng.uniform(-math.pi, math.pi), rng.choice([-1, 1])
    p1, p2 = spine + side * math.pi / 2, spine - side * math.pi / 2
    e1, e2 = math.radians(rng.uniform(300.0, 350.0)), math.radians(rng.uniform(300.0, 350.0))
    t1, t2 = p1 + rng.gauss(0.0, 0.3), p2 + rng.gauss(0.0, 0.3)
    a = _lobe(rng, (x[0] + math.cos(p1) * r1a, x[1] + math.sin(p1) * r1a), r1a, r1b, math.pi + (2.0 * math.pi - e1) / 2.0, e1, t1, c)
    b = _lobe(rng, (x[0] + math.cos(p2) * r2a, x[1] + math.sin(p2) * r2a), r2a, r2b, math.pi + (2.0 * math.pi - e2) / 2.0, e2, t2, c)
    return a + b[1:], x


def _fiber(rng, angle, c):
    S, m = c.size, c.margin
    if rng.random() < 0.8:
        start = (rng.uniform(m, S - m), rng.uniform(m, S - m))
    else:
        side, v = rng.choice(["top", "bottom", "left", "right"]), rng.uniform(m, S - m)
        start = {"top": (v, m), "bottom": (v, S - m), "left": (m, v), "right": (S - m, v)}[side]
    L = rng.uniform(0.40 * S, 0.95 * S)
    end = (_clamp(start[0] + math.cos(angle) * L, c), _clamp(start[1] + math.sin(angle) * L, c))
    rng.random()
    if rng.random() >= c.hairpin_p:
        return _wavy(rng, start, end, c.waviness, c)
    r, f = rng.uniform(*c.hairpin_r), rng.uniform(0.3, 0.7)
    e1 = (_clamp(start[0] + math.cos(angle) * (L * f), c), _clamp(start[1] + math.sin(angle) * (L * f), c))
    s1 = _wavy(rng, start, e1, c.waviness, c)
    hp, out = _hairpin(rng, e1, angle, r, c)
    L2, h = L * (1.0 - f) * rng.uniform(0.6, 1.0), hp[-1]
    s2 = _wavy(rng, h, (_clamp(h[0] + math.cos(out) * L2, c), _clamp(h[1] + math.sin(out) * L2, c)), c.waviness, c)
    return s1 + hp[1:] + s2[1:]


def _diverse_angle(rng, used, c):
    for _ in range(8):
        a = rng.uniform(-math.pi, math.pi)
        if not used or min(min(abs(a - u) % (2 * math.pi), 2 * math.pi - abs(a - u) % (2 * math.pi)) for u in used) > c.angle_sep:
            return a
    return rng.uniform(-math.pi, math.pi)


def _seg_dist(p1, p2, q1, q2):
    u, v, w = (p2[0] - p1[0], p2[1] - p1[1]), (q2[0] - q1[0], q2[1] - q1[1]), (p1[0] - q1[0], p1[1] - q1[1])
    a, b, cc = u[0] * u[0] + u[1] * u[1], u[0] * v[0] + u[1] * v[1], v[0] * v[0] + v[1] * v[1]
    d, e = u[0] * w[0] + u[1] * w[1], v[0] * w[0] + v[1] * w[1]
    den = a * cc - b * b
    if den > 1e-8:
        sc, tc = min(1.0, max(0.0, (b * e - cc * d) / den)), min(1.0, max(0.0, (a * e - b * d) / den))
    else:
        sc, tc = 0.0, min(1.0, max(0.0, (v[0] * w[0] + v[1] * w[1]) / cc if cc > 1e-8 else 0.0))
    return math.hypot(w[0] + sc * u[0] - tc * v[0], w[1] + sc * u[1] - tc * v[1])


def _conflict(P, Q, clearance, self_check=False):
    n = len(P)
    for s in range(0, len(Q), 1024):
        q = Q[s:s + 1024]
        a1x, a1y, a2x, a2y = (P[:, k, None] for k in range(4))
        b1x, b1y, b2x, b2y = (q[None, :, k] for k in range(4))
        o1 = (a2x - a1x) * (b1y - a1y) - (a2y - a1y) * (b1x - a1x)
        o2 = (a2x - a1x) * (b2y - a1y) - (a2y - a1y) * (b2x - a1x)
        o3 = (b2x - b1x) * (a1y - b1y) - (b2y - b1y) * (a1x - b1x)
        o4 = (b2x - b1x) * (a2y - b1y) - (b2y - b1y) * (a2x - b1x)
        hit = ((o1 == 0) & (o2 == 0) & (o3 == 0) & (o4 == 0)) | ((o1 * o2 <= 0) & (o3 * o4 <= 0))
        gx = np.maximum(np.minimum(a1x, a2x) - np.maximum(b1x, b2x), np.minimum(b1x, b2x) - np.maximum(a1x, a2x))
        gy = np.maximum(np.minimum(a1y, a2y) - np.maximum(b1y, b2y), np.minimum(b1y, b2y) - np.maximum(a1y, a2y))
        near = np.maximum(gx, 0) ** 2 + np.maximum(gy, 0) ** 2 < (clearance + 1e-6) ** 2
        if self_check:
            valid = np.arange(s, s + len(q))[None, :] >= np.arange(n)[:, None] + 2
            hit, near = hit & valid, near & valid
        if hit.any():
            return True
        for i, k in zip(*np.nonzero(near)):
            if _seg_dist(P[i, :2], P[i, 2:], q[k, :2], q[k, 2:]) < clearance:
                return True
    return False


def _segs(path):
    p = np.asarray(path, dtype=np.float64)
    return np.c_[p[:-1], p[1:]]


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return np.argwhere(x * x + y * y <= r * r) - r


def _pix(centers, r, size):
    yx = (np.unique(np.asarray(centers, dtype=np.int64).reshape(-1, 2), axis=0)[:, None] + _disk(r)[None]).reshape(-1, 2)
    yx = yx[(yx >= 0).all(1) & (yx < size).all(1)]
    return yx[:, 0], yx[:, 1]


def _draw(path, r, size):
    cs = []
    for p1, p2 in zip(path[:-1], path[1:]):
        steps = int(max(1, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))) + 1
        t = np.arange(steps) / max(steps - 1, 1)
        cs.append(np.c_[np.round(p1[1] + (p2[1] - p1[1]) * t), np.round(p1[0] + (p2[0] - p1[0]) * t)])
    return _pix(np.concatenate(cs), r, size)


def _raster(path, r, size):
    cs = []
    for p1, p2 in zip(path[:-1], path[1:]):
        steps = int(max(1, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))) + 1
        cs.append(np.c_[np.round(np.linspace(p1[1], p2[1], steps)), np.round(np.linspace(p1[0], p2[0], steps))])
    return _pix(np.concatenate(cs), r, size)


def _angles(paths, used):
    used.extend(math.atan2(p[-1][1] - p[0][1], p[-1][0] - p[0][0]) for p in paths if len(p) >= 2)


def make_mask(cls, seed, c):
    rng = random.Random(seed)
    canvas, coll = np.zeros((c.size, c.size), bool), np.zeros((c.size, c.size), bool)
    r, rj = max(1, c.thick // 2), max(2, c.thick // 2 + 1)
    target, used, segs, junction, added = rng.randint(*c.fibers), [], np.zeros((0, 4)), None, 0
    if cls != "negative":
        arms, junction = _fork(rng, cls, c)
        for a in arms:
            canvas[_draw(a, r, c.size)] = True
            coll[_raster(a, r, c.size)] = True
        jd = _pix([(junction[1], junction[0])], rj, c.size)
        canvas[jd] = coll[jd] = True
        segs = np.concatenate([_segs(a) for a in arms])
        _angles(arms, used)
        added += 1
    shapes = [g for g in (_p_shape, _eight) if c.p_loop > 0 and rng.random() < c.p_loop]
    for gen in shapes:
        for _ in range(c.attempts):
            path, x = gen(rng, c)
            if junction is not None and any(math.hypot(p[0] - junction[0], p[1] - junction[1]) < c.spacing * 8.0 for p in path):
                continue
            canvas[_draw(path, r, c.size)] = True
            canvas[_pix([(x[1], x[0])], rj, c.size)] = True
            _angles([path], used)
            added += 1
            break
    for _ in range(max(0, target - added)):
        angle = _diverse_angle(rng, used, c)
        for attempt in range(c.attempts):
            clearance = max(0.8, c.spacing * 0.9 ** (attempt // 15))
            path = _fiber(rng, angle, c)
            cand = _raster(path, r, c.size)
            if coll[cand].any():
                continue
            P = _segs(path)
            if _conflict(P, P, max(1.0, clearance * 0.2), self_check=True) or _conflict(P, segs, clearance):
                continue
            canvas[_draw(path, r, c.size)] = True
            coll[cand] = True
            segs = np.concatenate([segs, P])
            _angles([path], used)
            added += 1
            break
    return canvas, junction
