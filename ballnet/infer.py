#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from .models.ballnet import MultiViewBallNet
from .models.heads import ordinal_logits_to_probs, ordinal_expected_grade
from .models.heads import brand_with_unknown
from .data.dataset import imread_rgb, to_tensor, estimate_ball_lab_features, GRADE_ORDER


def load_folder(folder: str, views: int, img_size: int):
    paths = sorted(glob.glob(str(Path(folder) / "*.jpg"))) + sorted(glob.glob(str(Path(folder) / "*.png")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images in {folder}")
    if len(paths) < views:
        paths = (paths * (views // max(1, len(paths)) + 1))[:views]
    else:
        paths = paths[:views]
    imgs = [imread_rgb(p, img_size) for p in paths]
    Ls, bs = [], []
    for rgb in imgs:
        Lm, bm = estimate_ball_lab_features(rgb)
        Ls.append(Lm)
        bs.append(bm)
    aux = np.array([np.mean(Ls, dtype=np.float32), np.mean(bs, dtype=np.float32)], dtype=np.float32)
    x = torch.stack([to_tensor(im) for im in imgs], dim=0).unsqueeze(0)
    return x, torch.from_numpy(aux).unsqueeze(0), paths


def infer_folder(folder_path: str, weights: str, views: int = 6, img_size: int = 224,
                 backbone: str = "convnext_base", unknown_thr_brand: float | None = None,
                 unknown_thr_model: float | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights, map_location=device)
    brands = ckpt.get("brands", [])
    models = ckpt.get("models", [])
    n_brands = max(1, len(brands) or ckpt.get("args", {}).get("n_brands", 10))
    n_models = max(1, len(models) or ckpt.get("args", {}).get("n_models", 20))

    model = MultiViewBallNet(
        backbone_name=backbone,
        img_size=ckpt.get("args", {}).get("img_size", img_size),
        max_views=views,
        emb_dim=ckpt.get("args", {}).get("emb_dim", 256),
        n_grades=len(GRADE_ORDER),
        n_brands=n_brands,
        n_models=n_models,
        n_tags=ckpt.get("args", {}).get("n_tags", 7),
        aux_dim=2,
        pretrained=False,
    ).to(device)
    state = ckpt.get("model", ckpt)
    # Filter mismatched shapes (e.g., view_pos when views differ)
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    missing = set(model_state.keys()) - set(filtered.keys())
    if missing:
        pass
    model.load_state_dict(filtered, strict=False)
    model.eval()

    x, aux, _ = load_folder(folder_path, views, img_size)
    x = x.to(device); aux = aux.to(device)
    with torch.no_grad():
        out = model(x, aux)

    brand_logits = out["brand_logits"]
    model_logits = out["model_logits"]
    grade_probs = ordinal_logits_to_probs(out["grade_logits"].float())
    attw = out["attw"][0].detach().cpu().numpy().tolist()

    # Brand
    if unknown_thr_brand is not None:
        other_idx = brand_logits.shape[1] - 1
        brand_pred, brand_conf = brand_with_unknown(brand_logits, unknown_thr_brand, other_idx)
        bi = int(brand_pred[0]); bconf = float(brand_conf[0])
    else:
        bprob = torch.softmax(brand_logits, dim=1)
        bconf, bi_t = bprob.max(dim=1)
        bi = int(bi_t[0]); bconf = float(bconf[0])
    brand_probs = torch.softmax(brand_logits, dim=1)[0].detach().cpu().numpy().tolist()

    # Grade
    gi = int(grade_probs[0].argmax()); gconf = float(grade_probs[0, gi])
    grade_probs_dict = {GRADE_ORDER[i]: float(grade_probs[0, i]) for i in range(len(GRADE_ORDER))}

    # Map brand index to name if available
    brand_name = brands[bi] if brands and 0 <= bi < len(brands) else str(bi)
    brand_probs_named = { (brands[i] if i < len(brands) else str(i)) : float(brand_probs[i]) for i in range(len(brand_probs)) }

    return {
        "pred_grade": GRADE_ORDER[gi],
        "pred_grade_conf": gconf,
        "grade_probs": grade_probs_dict,
        "pred_brand": brand_name,
        "pred_brand_conf": bconf,
        "brand_probs": brand_probs_named,
        "att_weights": attw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=False, default=None)
    ap.add_argument("--ball-dir", type=str, required=True)
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--unknown-thr-brand", type=float, default=0.55)
    ap.add_argument("--unknown-thr-model", type=float, default=0.55)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiViewBallNet(
        backbone_name="convnext_base",
        img_size=args.img_size,
        max_views=args.views,
        emb_dim=256,
        n_grades=6,
        n_brands=10,
        n_models=20,
        n_tags=7,
        aux_dim=2,
        pretrained=False,
    ).to(device)
    model.eval()

    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)

    x, aux, paths = load_folder(args.ball_dir, args.views, args.img_size)
    x = x.to(device)
    aux = aux.to(device)

    with torch.no_grad():
        out = model(x, aux)
    brand_logits = out["brand_logits"]
    model_logits = out["model_logits"]
    grade_probs = ordinal_logits_to_probs(out["grade_logits"].float())
    exp_grade = ordinal_expected_grade(grade_probs)
    tag_probs = torch.sigmoid(out["tag_logits"]).squeeze(0)
    attw = out["attw"][0].detach().cpu().numpy()

    # Open-set routing (Other assumed last index)
    other_brand = brand_logits.shape[1] - 1
    other_model = model_logits.shape[1] - 1
    brand_pred, brand_conf = brand_with_unknown(brand_logits, args.unknown_thr_brand, other_brand)
    model_pred, model_conf = brand_with_unknown(model_logits, args.unknown_thr_model, other_model)

    # Top-k
    topk = min(args.topk, brand_logits.shape[1])
    brand_topk_conf, brand_topk_idx = torch.topk(torch.softmax(brand_logits, dim=1), k=topk, dim=1)
    model_topk_conf, model_topk_idx = torch.topk(torch.softmax(model_logits, dim=1), k=topk, dim=1)

    print("Brand top-k:")
    print(list(zip(brand_topk_idx[0].tolist(), brand_topk_conf[0].tolist())))
    print("Model top-k:")
    print(list(zip(model_topk_idx[0].tolist(), model_topk_conf[0].tolist())))
    print("Brand pred+conf (with Unknown routing):", int(brand_pred[0]), float(brand_conf[0]))
    print("Model pred+conf (with Unknown routing):", int(model_pred[0]), float(model_conf[0]))
    print("Grade distribution:", grade_probs[0].detach().cpu().numpy().round(3).tolist())
    print("Expected grade:", float(exp_grade[0]))
    print("Tag probs:", tag_probs.detach().cpu().numpy().round(3).tolist())
    print("Attention weights:", np.round(attw, 3).tolist())

    # If masks enabled, print % area per tag
    mask_logits = out.get("mask_logits")
    if mask_logits is not None:
        probs_mask = torch.sigmoid(mask_logits)[0]
        area = probs_mask.mean(dim=(1, 2))  # [n_tags]
        print("Mask area % per tag:", (area.detach().cpu().numpy() * 100.0).round(2).tolist())


if __name__ == "__main__":
    main()
