import os, glob, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


GRADE_ORDER = ["Mint", "Pearl", "A", "B", "C", "D"]
GRADE_TO_IDX = {g: i for i, g in enumerate(GRADE_ORDER)}


def imread_rgb(path: str, target_size: int = 224):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    img = img[y0 : y0 + m, x0 : x0 + m]
    if target_size and (img.shape[0] != target_size or img.shape[1] != target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    img = (img - mean) / std
    return torch.from_numpy(img)


def circle_mask(h, w, cx, cy, r):
    yy, xx = np.ogrid[:h, :w]
    dist = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist <= r * r


def estimate_ball_lab_features(rgb: np.ndarray) -> Tuple[float, float]:
    h, w = rgb.shape[:2]
    r = 0.45 * min(h, w)
    cy, cx = h / 2.0, w / 2.0
    mask = circle_mask(h, w, cx, cy, r)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) * (100.0 / 255.0)
    b = lab[..., 2].astype(np.float32) - 128.0
    return float(np.median(L[mask])), float(np.median(b[mask]))


class MultiViewBallDataset(Dataset):
    """
    Expects a structure:
    root/
      images/
        <ball_id>/view1.jpg ...
      labels.csv  with columns: ball_id, split, grade, brand, model, tags(optional json or comma list)

    Returns: (views[B,N,3,H,W], aux[B,2], targets dict)
    targets: grade idx, brand idx, model idx, tags (multi-hot)
    """

    def __init__(
        self,
        root: str,
        split: str,
        n_views: int = 8,
        img_size: int = 224,
        label_csv: str = "labels.csv",
        brands_json: Optional[str] = None,
        models_json: Optional[str] = None,
        allowed_brands: Optional[List[str]] = None,
        allowed_models: Optional[List[str]] = None,
        n_tags: int = 7,
    ):
        self.root = Path(root)
        self.images_root = self.root / "images"
        df = pd.read_csv(self.root / label_csv)
        self.df = df[df["split"].astype(str) == str(split)].copy()

        # Load brand/model vocabs if provided, else infer + add 'Other'
        brands = self._load_vocab(brands_json, key="brands") or sorted(
            [str(b) for b in self.df["brand"].fillna("Other").unique().tolist()]
        )
        models = self._load_vocab(models_json, key="models") or sorted(
            [str(m) for m in self.df["model"].fillna("Other").unique().tolist()]
        )
        if allowed_brands is not None:
            brands = allowed_brands
        if allowed_models is not None:
            models = allowed_models
        if "Other" not in brands:
            brands.append("Other")
        if "Other" not in models:
            models.append("Other")

        self.brands = brands
        self.models = models
        self.brand_to_idx = {b: i for i, b in enumerate(brands)}
        self.model_to_idx = {m: i for i, m in enumerate(models)}

        # Tagging: fixed vocabulary (7)
        self.n_tags = int(n_tags)

        self.samples = []
        for _, row in self.df.iterrows():
            ball_id = str(row["ball_id"]) if "ball_id" in row else str(row.get("id", ""))
            if not ball_id:
                continue
            grade = str(row.get("grade", "D"))
            brand = str(row.get("brand", "Other") or "Other")
            model = str(row.get("model", "Other") or "Other")
            folder = self.images_root / ball_id
            views = sorted(glob.glob(str(folder / "*.jpg"))) + sorted(glob.glob(str(folder / "*.png")))
            if len(views) == 0:
                continue

            # Parse tags (optional). Supported formats: JSON array, comma-separated, or missing
            tags_raw = row.get("tags", None)
            tags_vec = None
            if isinstance(tags_raw, str) and tags_raw.strip():
                try:
                    maybe = json.loads(tags_raw)
                    if isinstance(maybe, list):
                        tags_vec = np.zeros(self.n_tags, dtype=np.float32)
                        for j in maybe:
                            if isinstance(j, int) and 0 <= j < self.n_tags:
                                tags_vec[j] = 1.0
                except Exception:
                    parts = [p.strip() for p in tags_raw.split(",") if p.strip()]
                    tags_vec = np.zeros(self.n_tags, dtype=np.float32)
                    for p in parts:
                        try:
                            idx = int(p)
                            if 0 <= idx < self.n_tags:
                                tags_vec[idx] = 1.0
                        except Exception:
                            pass

            self.samples.append(
                {
                    "ball_id": ball_id,
                    "views": views,
                    "grade": grade,
                    "brand": brand if brand in self.brand_to_idx else "Other",
                    "model": model if model in self.model_to_idx else "Other",
                    "tags": tags_vec,
                }
            )

        self.split = split
        self.n_views = n_views
        self.img_size = img_size
        if len(self.samples) == 0:
            raise ValueError(f"No image folders under {self.images_root}")

    def _load_vocab(self, path: Optional[str], key: str) -> Optional[List[str]]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return [str(x) for x in obj]
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]
        except Exception:
            return None
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def _select_views(self, paths: List[str]) -> List[str]:
        if len(paths) >= self.n_views:
            return paths[: self.n_views]
        reps = []
        i = 0
        while len(reps) < self.n_views:
            reps.append(paths[i % len(paths)])
            i += 1
        return reps

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        paths = self._select_views(s["views"])
        imgs = [imread_rgb(p, self.img_size) for p in paths]
        Ls, bs = [], []
        for rgb in imgs:
            Lm, bm = estimate_ball_lab_features(rgb)
            Ls.append(Lm)
            bs.append(bm)
        aux = np.array([np.mean(Ls, dtype=np.float32), np.mean(bs, dtype=np.float32)], dtype=np.float32)
        views_t = torch.stack([to_tensor(im) for im in imgs], dim=0)
        grade_idx = GRADE_TO_IDX.get(str(s["grade"]), GRADE_TO_IDX["D"])  # default to toughest grade
        brand_idx = self.brand_to_idx.get(s["brand"], self.brand_to_idx["Other"])
        model_idx = self.model_to_idx.get(s["model"], self.model_to_idx["Other"])

        targets = {
            "grade": torch.tensor(grade_idx, dtype=torch.long),
            "brand": torch.tensor(brand_idx, dtype=torch.long),
            "model": torch.tensor(model_idx, dtype=torch.long),
        }
        if s.get("tags") is not None:
            targets["tags"] = torch.from_numpy(s["tags"])
        targets["ball_id"] = s["ball_id"]
        return views_t, torch.from_numpy(aux), targets

