import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import evaluate as ev

RES = Path(__file__).resolve().parent / "results"
MAIN = ["P", "R", "F1", "P_cls", "R_cls", "F1_cls"]
PER_CLASS = ["P_rep", "R_rep", "P_rev", "R_rev", "P_rep_cls", "R_rep_cls", "P_rev_cls", "R_rev_cls"]
NAMES = {"dino": "DINOv3 ViT-S/16 + MLP", "dinov2": "DINOv2 ViT-S/14 (registers) + MLP", "dino-ft4": "DINOv3 ViT-S/16 + MLP, last 4 blocks tuned", "dino-b": "DINOv3 ViT-B/16 + MLP", "frcnn": "Faster R-CNN R50-FPN v2"}
CLS_COLORS, MODEL_COLORS = ("#2a78d6", "#eb6834"), {"dino": "#2a78d6", "frcnn": "#eb6834", "dinov2": "#1baf7a"}
SHORT = {"dino": "DINOv3", "dinov2": "DINOv2", "frcnn": "Faster R-CNN"}
TP, FP, FN = "#0ca30c", "#d03b3b", "#fab219"


@lru_cache(None)
def _records(dataset):
    return data.records(dataset)


@lru_cache(None)
def load(name):
    cfg = json.load(open(RES / name / "metrics.json"))["cfg"]
    recs = _records(cfg["dataset"])
    gts = {s: {r["id"]: r["points"] for r in recs if r["split"] == s} for s in ("val", "test")}
    df = pd.read_csv(RES / name / "preds.csv.gz")
    preds = {s: ev.from_frame(df[df.split == s], gts[s]) for s in gts}
    return preds, gts, ev.best_threshold(preds["val"], gts["val"]), cfg


def runs(dataset):
    return sorted(p.parent.name for p in RES.glob(f"*_{dataset}_s*/metrics.json"))


def scores(dataset, split="test"):
    rows = []
    for name in runs(dataset):
        preds, gts, thr, cfg = load(name)
        rows.append(dict(model=name.rsplit(f"_{dataset}_s", 1)[0], seed=cfg["seed"], **ev.evaluate(preds[split], gts[split], thr)))
    return pd.DataFrame(rows)


def by_density(split="test"):
    groups = {r["id"]: r["group"] for r in _records("synthetic")}
    rows = []
    for name in runs("synthetic"):
        preds, gts, thr, cfg = load(name)
        g = ev.by_group(preds[split], gts[split], {k: groups[k] for k in gts[split]}, thr)
        rows.append(g.assign(model=name.rsplit("_synthetic_s", 1)[0], seed=cfg["seed"]).rename_axis("density").reset_index())
    return pd.concat(rows)


def summarize(df, cols, by="model"):
    rank = {k: i for i, k in enumerate(NAMES)}
    df = df.iloc[np.argsort([rank.get(m, len(rank)) for m in df.model], kind="stable")]
    df = df.assign(model=df.model.map(lambda k: NAMES.get(k, k)))
    g = df.groupby(by, sort=False)[cols]
    m, s, n = g.mean(), g.std().fillna(0), g.size()
    out = m.map("{:.2f}".format) + np.where(n.to_numpy()[:, None] > 1, " ± " + s.map("{:.2f}".format), "")
    return out.assign(n=n)


def to_latex(table):
    return "\n".join(f"{i} & " + " & ".join(r[c] for c in table.columns if c != "seeds") + r" \\" for i, r in table.iterrows()).replace(" ± ", r" $\pm$ ")


LABELS = {"P_cls": "P (cls)", "R_cls": "R (cls)", "F1_cls": "F1 (cls)", "P_rep": "rep P", "R_rep": "rep R", "P_rev": "rev P", "R_rev": "rev R",
          "P_rep_cls": "rep P (cls)", "R_rep_cls": "rep R (cls)", "P_rev_cls": "rev P (cls)", "R_rev_cls": "rev R (cls)", "n": "seeds"}


def write_tables():
    out = RES / "tables"
    out.mkdir(exist_ok=True)
    tabs = {}
    for d in ("real", "synthetic"):
        if runs(d):
            s = scores(d)
            s.to_csv(out / f"{d}_seeds.csv", index=False)
            tabs[f"{d}_main"], tabs[f"{d}_per_class"] = summarize(s, MAIN), summarize(s, PER_CLASS)
    if runs("synthetic"):
        dens = by_density()
        order = sorted(dens.density.unique(), key=lambda b: int(b[1:].split("-")[0]))
        t = summarize(dens, ["F1", "F1_cls"], by=["model", "density"]).drop(columns="n").unstack("model").reindex(order)
        t.columns = [f"{m} {LABELS.get(c, c)}" for c, m in t.columns]
        tabs["synthetic_density"] = t[sorted(t.columns, key=lambda c: (list(NAMES.values()).index(c.rsplit(" F1", 1)[0]) if c.rsplit(" F1", 1)[0] in NAMES.values() else 99, c))]
    for k, t in tabs.items():
        t = t.rename(columns=LABELS)
        (out / f"{k}.md").write_text(t.to_markdown())
        (out / f"{k}.tex").write_text(to_latex(t))
        tabs[k] = t
    return tabs


def show(name, ids=None, n=3, zoom=160):
    preds, gts, thr, cfg = load(name)
    recs = {r["id"]: r for r in _records(cfg["dataset"])}
    ids = ids or sorted(gts["test"], key=lambda k: -len(gts["test"][k]))[:n]
    fig, axes = plt.subplots(len(ids), 5, figsize=(20, 4 * len(ids)), squeeze=False)
    for ax, k in zip(axes, ids):
        img, g, p = data.load_image(recs[k]["path"]), gts["test"][k], preds["test"][k]
        p = p[p[:, 3] >= thr]
        m = ev.match(p, g)
        tp, fp = [i for i in range(len(p)) if i in set(m[:, 0])], [i for i in range(len(p)) if i not in set(m[:, 0])]
        fn = [j for j in range(len(g)) if j not in set(m[:, 1])]
        s = data.SCALE
        ax[0].imshow(img, cmap="gray")
        ax[0].scatter(g[:, 0] * s, g[:, 1] * s, marker="x", s=90, linewidths=2, c=[CLS_COLORS[int(c)] for c in g[:, 2]])
        ax[0].scatter(p[tp, 0] * s, p[tp, 1] * s, marker="o", s=300, facecolors="none", edgecolors=TP, linewidths=2)
        ax[0].scatter(p[fp, 0] * s, p[fp, 1] * s, marker="s", s=300, facecolors="none", edgecolors=FP, linewidths=2)
        ax[0].set_title(f"{k[:42]}\nTP {len(tp)}  FP {len(fp)}  FN {len(fn)}", fontsize=9)
        crops = [("TP", p[i], TP, "o") for i in tp] + [("FN", g[j], FN, "o") for j in fn] + [("FP", p[i], FP, "s") for i in fp]
        for a, (tag, q, col, mk) in zip(ax[1:], crops[:4]):
            x, y = int(q[0] * s), int(q[1] * s)
            x0, y0 = np.clip([x - zoom, y - zoom], 0, np.array(img.shape[::-1]) - 2 * zoom)
            a.imshow(img[y0:y0 + 2 * zoom, x0:x0 + 2 * zoom], cmap="gray")
            a.scatter([x - x0], [y - y0], marker=mk, s=900, facecolors="none", edgecolors=col, linewidths=2, linestyles="--" if tag == "FN" else "-")
            a.set_title(f"{tag}: {'GT ' if tag == 'FN' else 'pred '}{data.CLASSES[int(q[2])]}" + (f" ({q[3]:.2f})" if tag != "FN" else ""), fontsize=9)
        for a in ax:
            a.axis("off")
    fig.suptitle(f"{NAMES.get(name.rsplit('_', 2)[0], name)} | {name} | threshold {thr:.2f}\nx = ground truth (blue replication, orange reversed); circle = true positive, square = false positive, dashed = missed", fontsize=11)
    fig.tight_layout()
    return fig


def pr_curve(dataset, split="test", ax=None):
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    seen = set()
    for name in runs(dataset):
        model = name.rsplit(f"_{dataset}_s", 1)[0]
        if model not in MODEL_COLORS:
            continue
        preds, gts, thr, _ = load(name)
        m = [x for t in np.arange(0.02, 1, 0.02) if (x := ev.evaluate(preds[split], gts[split], t))["n_pred"] > 0]
        ax.plot([x["R"] for x in m], [x["P"] for x in m], color=MODEL_COLORS[model], lw=1.5, alpha=0.8, label=None if model in seen else NAMES[model])
        o = ev.evaluate(preds[split], gts[split], thr)
        ax.plot(o["R"], o["P"], "o", color=MODEL_COLORS[model], ms=8, mec="white", mew=1.5)
        seen.add(model)
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1), title=f"{dataset} {split}, class-agnostic (dot = val-selected threshold)")
    ax.grid(color="#e8e7e3", lw=0.8)
    ax.legend(fontsize=8, frameon=False)
    return ax


def density_plot(metric="F1", ax=None):
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    d = by_density()
    order = sorted(d.density.unique(), key=lambda b: int(b[1:].split("-")[0]))
    for model, g in d.groupby("model"):
        if model not in MODEL_COLORS:
            continue
        st = g.groupby("density")[metric].agg(["mean", "std"]).reindex(order)
        x = np.arange(len(order))
        ax.errorbar(x, st["mean"], yerr=st["std"].fillna(0), color=MODEL_COLORS[model], lw=2, marker="o", ms=8, capsize=3, label=NAMES[model])
        ax.annotate(SHORT[model], (x[-1], st["mean"].iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color="#52514e")
    ax.set_xticks(np.arange(len(order)), [b.replace("f", "") for b in order])
    ax.set_xlim(-0.5, len(order) + 0.6)
    ax.set(xlabel="fibers per image", ylabel=f"{metric} ({'class-dependent' if 'cls' in metric else 'class-agnostic'})", ylim=(0, 1), title="Synthetic test set: detection vs. scene density")
    ax.grid(axis="y", color="#e8e7e3", lw=0.8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return ax


if __name__ == "__main__":
    for k, t in write_tables().items():
        print(k, t.to_markdown(), sep="\n")
