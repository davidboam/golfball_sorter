#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .models.ballnet import MultiViewBallNet
from .models.heads import OrdinalLoss
from .utils.optim import EMA, maybe_autocast
from .utils.metrics import dice_loss
from .data.dataset import MultiViewBallDataset


def fake_batch(B: int = 2, N: int = 8, H: int = 224, W: int = 224, n_grades: int = 6, n_brands: int = 5, n_models: int = 7, n_tags: int = 7):
    views = torch.randn(B, N, 3, H, W)
    aux = torch.randn(B, 2)
    y_grade = torch.randint(low=0, high=n_grades, size=(B,))
    y_brand = torch.randint(low=0, high=n_brands, size=(B,))
    y_model = torch.randint(low=0, high=n_models, size=(B,))
    y_tags = (torch.rand(B, n_tags) > 0.7).float()
    y_masks = (torch.rand(B, n_tags, H // 4, W // 4) > 0.8).float()
    return views, aux, y_grade, y_brand, y_model, y_tags, y_masks


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--label-csv", type=str, default="labels.csv")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--backbone", type=str, default="convnext_base", choices=["convnext_base", "vit_base_patch16_224"]) 
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--brands-json", type=str, default=None)
    ap.add_argument("--models-json", type=str, default=None)
    ap.add_argument("--use-netvlad", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Model sizes inferred from vocabs (smoke uses small values)
    n_grades = 6
    n_brands = 8
    n_models = 12
    n_tags = 7

    model = MultiViewBallNet(
        backbone_name=args.backbone,
        img_size=args.img_size,
        max_views=args.views,
        emb_dim=256,
        n_grades=n_grades,
        n_brands=n_brands,
        n_models=n_models,
        n_tags=n_tags,
        aux_dim=2,
        use_netvlad=args.use_netvlad,
        pretrained=False,
    ).to(device)

    loss_grade = OrdinalLoss(n_grades)
    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(model, decay=0.999) if args.ema else None

    # In this refactor, we focus on smoke-test path. Data loader wiring can be added later.
    if args.smoke_test:
        model.train()
        views, aux, y_grade, y_brand, y_model, y_tags, y_masks = fake_batch(
            B=args.batch_size, N=args.views, H=args.img_size, W=args.img_size, n_grades=n_grades, n_brands=n_brands, n_models=n_models, n_tags=n_tags
        )
        views = views.to(device)
        aux = aux.to(device)
        y_grade = y_grade.to(device)
        y_brand = y_brand.to(device)
        y_model = y_model.to(device)
        y_tags = y_tags.to(device)
        y_masks = y_masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(args.amp):
            out = model(views, aux)
            lg = out["grade_logits"]
            lb = out["brand_logits"]
            lm = out["model_logits"]
            lt = out["tag_logits"]
            lmask = out["mask_logits"]

            Lg = loss_grade(lg, y_grade)
            Lb = loss_ce(lb, y_brand)
            Lm = loss_ce(lm, y_model)
            Lt = loss_bce(lt, y_tags)
            # Resize target masks to match stub output
            y_masks_resized = torch.nn.functional.interpolate(y_masks, size=lmask.shape[-2:], mode="nearest")
            Lmask = dice_loss(lmask, y_masks_resized)

            # Consistency: if large defect area but high grade, penalize
            probs_mask = torch.sigmoid(lmask)
            area = probs_mask.mean(dim=(2, 3))  # [B, n_tags]
            large_defect = area.mean(dim=1)  # [B]
            high_grade = (y_grade.float() <= 1.0).float()  # Mint/Pearl
            Lcons = (large_defect * high_grade).mean()

            loss_total = Lg + Lb + 0.5 * Lm + 0.2 * Lt + 0.2 * Lmask + 0.1 * Lcons

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        eg = out["grade_logits"].detach()
        # Report
        print({
            "loss_total": float(loss_total.detach().cpu()),
            "loss_grade": float(Lg.detach().cpu()),
            "loss_brand": float(Lb.detach().cpu()),
            "loss_model": float(Lm.detach().cpu()),
            "loss_tags": float(Lt.detach().cpu()),
            "loss_masks": float(Lmask.detach().cpu()),
            "loss_cons": float(Lcons.detach().cpu()),
        })

        # Expected grade and top-1 brand
        from .models.heads import ordinal_logits_to_probs, ordinal_expected_grade

        probs_g = ordinal_logits_to_probs(out["grade_logits"].float())
        eg = ordinal_expected_grade(probs_g)
        brand_probs = torch.softmax(out["brand_logits"], dim=1)
        brand_conf, brand_pred = brand_probs.max(dim=1)
        print("expected_grade:", eg.detach().cpu().numpy().tolist())
        print("brand_pred:", brand_pred.cpu().numpy().tolist(), "conf:", brand_conf.detach().cpu().numpy().tolist())
        if out.get("attw") is not None:
            print("attn_weights_sample:", out["attw"][0].detach().cpu().numpy().round(3).tolist())

        if args.save:
            ckpt = {
                "model": model.state_dict(),
                "args": {
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "views": args.views,
                    "emb_dim": 256,
                    "n_grades": n_grades,
                    "n_brands": n_brands,
                    "n_models": n_models,
                    "n_tags": n_tags,
                },
                # Smoke test has no real vocabularies
                "brands": [f"Brand{i}" for i in range(n_brands)],
                "models": [f"Model{i}" for i in range(n_models)],
            }
            torch.save(ckpt, args.save)
            print("saved:", args.save)
        return

    # Real training path
    print("Building dataset and starting training...")
    try:
        ds = MultiViewBallDataset(
            root=args.root,
            split=args.split,
            n_views=args.views,
            img_size=args.img_size,
            label_csv=args.label_csv,
            brands_json=args.brands_json,
            models_json=args.models_json,
        )
    except Exception as e:
        print("Dataset error:", e)
        print("Hint: ensure structure root/images/<ball_id>/*.jpg and labels.csv with columns: ball_id, split, grade, brand, model, [tags]")
        return

    n_grades = 6
    n_brands = len(ds.brands)
    n_models = len(ds.models)
    n_tags = ds.n_tags

    # Collate to keep ball_id strings safely
    def make_collate(n_tags):
        def collate(batch):
            views = torch.stack([b[0] for b in batch], dim=0)
            aux = torch.stack([b[1] for b in batch], dim=0)
            tlist = [b[2] for b in batch]
            targets = {
                "grade": torch.stack([t["grade"] for t in tlist], dim=0),
                "brand": torch.stack([t["brand"] for t in tlist], dim=0),
                "model": torch.stack([t["model"] for t in tlist], dim=0),
                "ball_id": [t["ball_id"] for t in tlist],
            }
            if "tags" in tlist[0]:
                tags = []
                for t in tlist:
                    v = t.get("tags")
                    if v is None:
                        v = torch.zeros(n_tags, dtype=torch.float32)
                    tags.append(v)
                targets["tags"] = torch.stack(tags, dim=0)
            return views, aux, targets
        return collate

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=make_collate(n_tags)
    )

    model = MultiViewBallNet(
        backbone_name=args.backbone,
        img_size=args.img_size,
        max_views=args.views,
        emb_dim=256,
        n_grades=n_grades,
        n_brands=n_brands,
        n_models=n_models,
        n_tags=n_tags,
        aux_dim=2,
        use_netvlad=args.use_netvlad,
        pretrained=False,
    ).to(device)

    loss_grade = OrdinalLoss(n_grades)
    loss_ce = nn.CrossEntropyLoss()
    loss_bce = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(model, decay=0.999) if args.ema else None

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for it, (views, aux, tgt) in enumerate(dl):
            views = views.to(device)
            aux = aux.to(device)
            y_grade = tgt["grade"].to(device)
            y_brand = tgt["brand"].to(device)
            y_model = tgt["model"].to(device)
            y_tags = tgt.get("tags")
            if y_tags is not None:
                y_tags = y_tags.to(device)

            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast(args.amp):
                out = model(views, aux)
                lg = out["grade_logits"]
                lb = out["brand_logits"]
                lm = out["model_logits"]
                lt = out["tag_logits"]

                Lg = loss_grade(lg, y_grade)
                Lb = loss_ce(lb, y_brand)
                Lm = loss_ce(lm, y_model)
                Lt = loss_bce(lt, y_tags) if y_tags is not None else 0.0 * lg.sum()
                # Segmentation targets not provided -> stub zero loss
                Lmask = 0.0 * lg.sum()

                # Consistency stub: same as smoke-test but using predictions
                probs_mask = torch.sigmoid(out["mask_logits"])  # [B,n_tags,H',W']
                area = probs_mask.mean(dim=(2, 3))  # [B, n_tags]
                large_defect = area.mean(dim=1)
                high_grade = (y_grade.float() <= 1.0).float()
                Lcons = (large_defect * high_grade).mean()

                loss_total = Lg + Lb + 0.5 * Lm + 0.2 * Lt + 0.1 * Lcons + 0.0 * Lmask

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            total += float(loss_total.detach().cpu())
            if (it + 1) % 10 == 0:
                print(f"epoch {epoch+1}/{args.epochs} iter {it+1} avg_loss {total/(it+1):.4f}")

        if args.save:
            out_path = args.save if args.epochs == 1 else f"{Path(args.save).stem}_e{epoch+1}.pth"
            ckpt = {
                "model": model.state_dict(),
                "args": {
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "views": args.views,
                    "emb_dim": 256,
                    "n_grades": n_grades,
                    "n_brands": n_brands,
                    "n_models": n_models,
                    "n_tags": n_tags,
                    "brands_json": args.brands_json,
                    "models_json": args.models_json,
                },
                "brands": ds.brands,
                "models": ds.models,
            }
            torch.save(ckpt, out_path)
            print("saved:", out_path)

    print("Training complete.")


if __name__ == "__main__":
    main()
