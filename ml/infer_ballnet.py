#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_ballnet import MultiViewBallNet, ordinal_logits_to_probs, GRADE_ORDER, imread_rgb, to_tensor, estimate_ball_lab_features

def load_images(folder: Path, n_views: int, img_size: int = 224):
    paths = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))[:n_views]
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under {folder}")
    while len(paths) < n_views:
        paths.append(paths[-1])
    imgs = [imread_rgb(str(p), img_size) for p in paths]
    auxL, auxb = [], []
    for im in imgs:
        Lm, bm = estimate_ball_lab_features(im)
        auxL.append(Lm); auxb.append(bm)
    aux = np.array([np.mean(auxL), np.mean(auxb)], dtype=np.float32)
    views = torch.stack([to_tensor(im) for im in imgs], dim=0)  # [N,C,H,W]
    return views.unsqueeze(0), torch.from_numpy(aux).unsqueeze(0)

def infer_folder(folder_path: str, ckpt_path: str, n_views: int = 4, backbone: str = "mobilenetv3_small_100"):
    folder = Path(folder_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    brands = ckpt.get("brands", ckpt.get("brand_list", []))
    if not brands: raise ValueError("Brands list not found in checkpoint.")
    n_brands = len(brands)
    model = MultiViewBallNet(backbone_name=backbone, n_grades=len(GRADE_ORDER), n_brands=n_brands, emb_dim=ckpt["args"].get("emb_dim", 256), aux_dim=2)
    model.load_state_dict(ckpt["model"], strict=True); model.eval()
    views, aux = load_images(folder, n_views)
    with torch.no_grad():
        glog, blog, attw = model(views, aux)
        p_grade = ordinal_logits_to_probs(glog)[0].numpy()
        p_brand = F.softmax(blog, dim=1)[0].numpy()
        att = attw[0].numpy()
    grade_idx = int(p_grade.argmax()); brand_idx = int(p_brand.argmax())
    return {"pred_grade": GRADE_ORDER[grade_idx], "pred_grade_conf": float(p_grade[grade_idx]),
            "grade_probs": {GRADE_ORDER[i]: float(p_grade[i]) for i in range(len(GRADE_ORDER))},
            "pred_brand": brands[brand_idx], "pred_brand_conf": float(p_brand[brand_idx]),
            "brand_probs": {brands[i]: float(p_brand[i]) for i in range(len(brands))},
            "att_weights": [float(x) for x in att.tolist()]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    args = ap.parse_args()
    print(json.dumps(infer_folder(args.images, args.ckpt, n_views=args.n_views, backbone=args.backbone), indent=2))

if __name__ == "__main__":
    main()
