import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import data
import evaluate as ev


def build(size):
    m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=size, max_size=size, box_score_thresh=0.01, box_detections_per_img=100)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 3)
    return m


def target(p, box, size):
    b = np.clip(np.c_[p[:, :2] - box / 2, p[:, :2] + box / 2], 0, size).reshape(-1, 4)
    return {"boxes": torch.tensor(b, dtype=torch.float32).cuda(), "labels": torch.tensor(p[:, 2] + 1, dtype=torch.int64).cuda()}


def to_input(img):
    return torch.from_numpy(img).cuda().float().div(255)[None].expand(3, -1, -1)


@torch.no_grad()
def predict(model, recs, imgs):
    model.eval()
    out = {}
    for r, im in zip(recs, imgs):
        model.transform.min_size, model.transform.max_size = (max(im.shape),), max(im.shape)
        d = model([to_input(im)])[0]
        c = (d["boxes"][:, :2] + d["boxes"][:, 2:]) / 2 / data.SCALE
        out[r["id"]] = ev.point_nms(torch.cat([c, d["labels"][:, None] - 1.0, d["scores"][:, None]], 1).cpu().numpy())
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="real", choices=["real", "synthetic"])
    pa.add_argument("--crop", type=int, default=1024)
    pa.add_argument("--box", type=float, default=64)
    pa.add_argument("--steps", type=int, default=6000)
    pa.add_argument("--batch", type=int, default=4)
    pa.add_argument("--lr", type=float, default=0.01)
    pa.add_argument("--pos_frac", type=float, default=0.5)
    pa.add_argument("--eval_every", type=int, default=500)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--name", default=None)
    a = pa.parse_args()
    name = a.name or f"frcnn_{a.dataset}_s{a.seed}"
    out = Path(__file__).resolve().parent / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = data.DATA / "checkpoints" / f"{name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    S = {s: data.load_split(a.dataset, s) for s in ("train", "val", "test")}
    gts = {s: {r["id"]: r["points"] for r in S[s][0]} for s in S}
    model = build(a.crop).cuda()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=a.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    _, imgs, pts = S["train"]
    best = -1.0
    for step in range(1, a.steps + 1):
        model.train()
        model.transform.min_size, model.transform.max_size = (a.crop,), a.crop
        crops = [data.sample_crop(imgs, pts, a.crop, a.pos_frac, rng) for _ in range(a.batch)]
        x = data.jitter(torch.from_numpy(np.stack([c[0] for c in crops])).cuda().float().div(255)[:, None]).expand(-1, 3, -1, -1)
        loss = sum(model(list(x), [target(c[1], a.box, a.crop) for c in crops]).values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            pv = predict(model, *S["val"][:2])
            m = ev.evaluate(pv, gts["val"], ev.best_threshold(pv, gts["val"]))
            print(f"step {step} loss {loss.item():.4f} val F1 {m['F1']:.3f} F1_cls {m['F1_cls']:.3f} thr {m['thr']:.2f}", flush=True)
            if m["F1"] > best:
                best = m["F1"]
                torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    preds = {s: predict(model, *S[s][:2]) for s in ("val", "test")}
    m = ev.save_run(out, preds, gts, **vars(a))
    print(f"test P {m['P']:.3f} R {m['R']:.3f} F1 {m['F1']:.3f} | cls P {m['P_cls']:.3f} R {m['R_cls']:.3f} F1 {m['F1_cls']:.3f}")


if __name__ == "__main__":
    main()
