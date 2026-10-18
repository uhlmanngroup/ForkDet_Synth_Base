import argparse
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
import evaluate as ev


class DinoDet(nn.Module):
    def __init__(self, name, unfreeze=0, hidden=1024, drop=0.2, n_cls=2):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0, **({"dynamic_img_size": True} if "dinov2" in name else {}))
        self.backbone.requires_grad_(False)
        self.tune = unfreeze > 0
        for m in (list(self.backbone.blocks[-unfreeze:]) + [self.backbone.norm]) if self.tune else []:
            m.requires_grad_(True)
        self.patch = self.backbone.patch_embed.patch_size[0]
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(self.backbone.num_features, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, n_cls))
        nn.init.constant_(self.head[-1].bias, -4.6)

    def forward(self, x):
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        with torch.set_grad_enabled(self.training and self.tune):
            t = self.backbone.forward_features(x)[:, self.backbone.num_prefix_tokens:]
        return self.head(t).transpose(1, 2).reshape(len(x), -1, h, w)


def heatmap(p, n, patch, sigma):
    hm = np.zeros((2, n, n), np.float32)
    c = (np.arange(n) + 0.5) * patch
    for x, y, k in p:
        g = np.exp(-((c[None] - x) ** 2 + (c[:, None] - y) ** 2) / (2 * (sigma * patch) ** 2))
        g[min(int(y // patch), n - 1), min(int(x // patch), n - 1)] = 1
        hm[int(k)] = np.maximum(hm[int(k)], g)
    return hm


def focal(logit, hm):
    p = logit.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = hm.eq(1).float()
    loss = -(pos * (1 - p) ** 2 * p.log() + (1 - pos) * (1 - hm) ** 4 * p ** 2 * (1 - p).log())
    return loss.sum() / pos.sum().clamp(min=1)


@torch.no_grad()
def heat(model, img, n, bs=32):
    P, (H, W) = model.patch, img.shape
    h, w = -(-H // P), -(-W // P)
    x = F.pad(torch.from_numpy(img).cuda().float().div(255)[None, None], (0, w * P - W, 0, h * P - H))
    pos = lambda L: sorted(set(range(0, max(L - n, 0) + 1, n * 3 // 4)) | {max(L - n, 0)})
    tiles = [(i, j) for i in pos(h) for j in pos(w)]
    out, cnt = torch.zeros(2, h, w, device="cuda"), torch.zeros(1, h, w, device="cuda")
    for b in range(0, len(tiles), bs):
        ts = tiles[b:b + bs]
        with torch.autocast("cuda", torch.bfloat16):
            y = model(torch.cat([x[..., i * P:(i + n) * P, j * P:(j + n) * P] for i, j in ts])).float().sigmoid()
        for (i, j), yi in zip(ts, y):
            out[:, i:i + n, j:j + n] += yi
            cnt[:, i:i + n, j:j + n] += 1
    return out / cnt


def peaks(hm, patch, scale, min_score=0.02):
    m, c = hm.max(0)
    iy, ix = ((m == F.max_pool2d(m[None, None], 3, 1, 1)[0, 0]) & (m >= min_score)).nonzero(as_tuple=True)
    p = torch.stack([(ix + 0.5) * patch / scale, (iy + 0.5) * patch / scale, c[iy, ix].float(), m[iy, ix]], 1)
    return ev.point_nms(p.cpu().numpy())


def predict(model, recs, imgs, n):
    model.eval()
    return {r["id"]: peaks(heat(model, im, n), model.patch, data.SCALE) for r, im in zip(recs, imgs)}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="real", choices=["real", "synthetic"])
    pa.add_argument("--backbone", default="vit_small_patch16_dinov3.lvd1689m")
    pa.add_argument("--unfreeze", type=int, default=0)
    pa.add_argument("--tokens", type=int, default=32)
    pa.add_argument("--steps", type=int, default=4000)
    pa.add_argument("--batch", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--lr_backbone", type=float, default=2e-5)
    pa.add_argument("--pos_frac", type=float, default=0.5)
    pa.add_argument("--sigma", type=float, default=1.0)
    pa.add_argument("--eval_every", type=int, default=250)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--name", default=None)
    pa.add_argument("--predict", action="store_true")
    a = pa.parse_args()
    name = a.name or f"dino_{a.dataset}_s{a.seed}"
    out = Path(__file__).resolve().parent / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = data.DATA / "checkpoints" / f"{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    S = {s: data.load_split(a.dataset, s) for s in (("val", "test") if a.predict else ("train", "val", "test"))}
    gts = {s: {r["id"]: r["points"] for r in S[s][0]} for s in S}
    model = DinoDet(a.backbone, a.unfreeze).cuda()
    groups = [{"params": model.head.parameters(), "lr": a.lr}]
    if model.tune:
        groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": a.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.steps, pct_start=0.1)
    tile, best = a.tokens * model.patch, -1.0
    for step in range(1, 0 if a.predict else a.steps + 1):
        _, imgs, pts = S["train"]
        model.train()
        crops = [data.sample_crop(imgs, pts, tile, a.pos_frac, rng) for _ in range(a.batch)]
        x = data.jitter(torch.from_numpy(np.stack([c[0] for c in crops])).cuda().float().div(255)[:, None])
        y = torch.from_numpy(np.stack([heatmap(c[1], a.tokens, model.patch, a.sigma) for c in crops])).cuda()
        with torch.autocast("cuda", torch.bfloat16):
            loss = focal(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            pv = predict(model, *S["val"][:2], a.tokens)
            m = ev.evaluate(pv, gts["val"], ev.best_threshold(pv, gts["val"]))
            print(f"step {step} loss {loss.item():.4f} val F1 {m['F1']:.3f} F1_cls {m['F1_cls']:.3f} thr {m['thr']:.2f}", flush=True)
            if m["F1"] > best:
                best = m["F1"]
                torch.save({k: v for k, v in model.state_dict().items() if k.startswith("head") or model.tune}, ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
    preds = {s: predict(model, *S[s][:2], a.tokens) for s in ("val", "test")}
    m = ev.save_run(out, preds, gts, **vars(a))
    print(f"test P {m['P']:.3f} R {m['R']:.3f} F1 {m['F1']:.3f} | cls P {m['P_cls']:.3f} R {m['R_cls']:.3f} F1 {m['F1_cls']:.3f}")


if __name__ == "__main__":
    main()
