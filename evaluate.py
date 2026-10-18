import json

import numpy as np
import pandas as pd

MATCH_PX = 75.0
NAMES = ("rep", "rev")


def match(pred, gt, thr=MATCH_PX):
    if not len(pred) or not len(gt):
        return np.zeros((0, 2), int)
    d = np.linalg.norm(pred[:, None, :2] - gt[None, :, :2], axis=-1)
    i, j = np.nonzero(d <= thr)
    used_i, used_j, out = set(), set(), []
    for k in np.argsort(d[i, j], kind="stable"):
        if i[k] not in used_i and j[k] not in used_j:
            used_i.add(i[k]); used_j.add(j[k]); out.append((i[k], j[k]))
    return np.array(out, int).reshape(-1, 2)


def point_nms(p, radius=MATCH_PX):
    p = p[np.argsort(-p[:, 3])]
    near = np.linalg.norm(p[:, None, :2] - p[None, :, :2], axis=-1) <= radius
    keep = np.ones(len(p), bool)
    for k in range(len(p)):
        if keep[k]:
            keep[k + 1:] &= ~near[k, k + 1:]
    return p[keep]


def _div(a, b):
    return a / b if b else 0.0


def _f1(p, r):
    return _div(2 * p * r, p + r)


def evaluate(preds, gts, thr=0.0):
    c = {}
    add = lambda k, v: c.__setitem__(k, c.get(k, 0) + int(v))
    for k, g in gts.items():
        p = preds.get(k, np.zeros((0, 4)))
        p = p[p[:, 3] >= thr]
        m = match(p, g)
        pc, gc = p[m[:, 0], 2], g[m[:, 1], 2]
        add("n_pred", len(p)); add("n_gt", len(g)); add("tp", len(m)); add("tp_cls", (pc == gc).sum())
        for i, n in enumerate(NAMES):
            add(f"pred_{n}", (p[:, 2] == i).sum()); add(f"gt_{n}", (g[:, 2] == i).sum())
            add(f"tp_pred_{n}", (pc == i).sum()); add(f"tp_gt_{n}", (gc == i).sum()); add(f"tp_cls_{n}", ((pc == i) & (gc == i)).sum())
    P, R = _div(c["tp"], c["n_pred"]), _div(c["tp"], c["n_gt"])
    Pc, Rc = _div(c["tp_cls"], c["n_pred"]), _div(c["tp_cls"], c["n_gt"])
    out = dict(thr=thr, P=P, R=R, F1=_f1(P, R), P_cls=Pc, R_cls=Rc, F1_cls=_f1(Pc, Rc))
    for n in NAMES:
        out.update({f"P_{n}": _div(c[f"tp_pred_{n}"], c[f"pred_{n}"]), f"R_{n}": _div(c[f"tp_gt_{n}"], c[f"gt_{n}"]),
                    f"P_{n}_cls": _div(c[f"tp_cls_{n}"], c[f"pred_{n}"]), f"R_{n}_cls": _div(c[f"tp_cls_{n}"], c[f"gt_{n}"])})
    return {**out, **c}


def best_threshold(preds, gts, grid=np.round(np.arange(0.02, 1.0, 0.02), 2)):
    return max(grid, key=lambda t: (evaluate(preds, gts, t)["F1"], -t))


def by_group(preds, gts, groups, thr):
    rows = {g: evaluate(preds, {k: v for k, v in gts.items() if groups[k] == g}, thr) for g in sorted(set(groups.values()), key=lambda s: int(s[1:].split("-")[0]) if s[1:2].isdigit() else s)}
    return pd.DataFrame(rows).T


def to_frame(preds):
    return pd.DataFrame([(k, x, y, int(c), s) for k, p in preds.items() for x, y, c, s in p], columns=["image", "x", "y", "cls", "score"])


def from_frame(df, ids):
    return {k: df[df.image == k][["x", "y", "cls", "score"]].to_numpy(np.float32) for k in ids}


def save_run(out, preds, gts, **cfg):
    pd.concat([to_frame(preds[s]).assign(split=s) for s in preds]).to_csv(out / "preds.csv.gz", index=False)
    thr = best_threshold(preds["val"], gts["val"])
    m = {s: evaluate(preds[s], gts[s], thr) for s in preds}
    json.dump(dict(cfg=cfg, **m), open(out / "metrics.json", "w"), indent=1, default=float)
    return m["test"]
