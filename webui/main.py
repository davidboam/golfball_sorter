#!/usr/bin/env python3
import os, json, threading, shutil
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks, Form, Body, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
import numpy

BASE = Path(__file__).resolve().parents[1]
# Ensure repo root is first on sys.path so package imports resolve to local modules
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from capture.server import Server, accept_loop, SAVE_DIR
from ballnet.infer import infer_folder
from ballnet.data.dataset import imread_rgb, to_tensor, estimate_ball_lab_features, GRADE_ORDER
from ballnet.models.ballnet import MultiViewBallNet
from utils.gradcam import gradcam_for_view, overlay_heatmap
import torch

# cache last label for sticky UI defaults
LAST_LABEL = {"brand": "", "grade": "", "color": "", "model": ""}
# ---- Model cache ----
MODEL_CACHE = {"model": None, "brands": None, "args": None}

def load_model_from_ckpt(ckpt_path: str):
    if not ckpt_path:
        return None
    import torch
    from ballnet.models.ballnet import MultiViewBallNet
    from ballnet.data.dataset import GRADE_ORDER
    ckpt = torch.load(ckpt_path, map_location="cpu")
    brands = ckpt.get("brands", [])
    models = ckpt.get("models", [])
    args = ckpt.get("args", {})
    backbone = args.get("backbone", "convnext_base")
    emb_dim = args.get("emb_dim", 256)
    img_size = args.get("img_size", 224)
    views = args.get("views", int(os.getenv("N_VIEWS", "4")))
    n_brands = len(brands) if brands else args.get("n_brands", 10)
    n_models = len(models) if models else args.get("n_models", 20)
    model = MultiViewBallNet(
        backbone_name=backbone,
        img_size=img_size,
        max_views=views,
        emb_dim=emb_dim,
        n_grades=len(GRADE_ORDER),
        n_brands=n_brands,
        n_models=n_models,
        aux_dim=2,
        pretrained=False,
    )
    # Robust load: filter mismatched shapes
    state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, (brands or None), args

def ensure_model_loaded():
    if MODEL_CACHE["model"] is None and MODEL_CKPT:
        res = load_model_from_ckpt(MODEL_CKPT)
        if res:
            m, b, a = res
            MODEL_CACHE["model"] = m
            MODEL_CACHE["brands"] = b
            MODEL_CACHE["args"] = a
    return MODEL_CACHE["model"] is not None


APP_HOST = os.getenv("APP_HOST", "0.0.0.0"); APP_PORT = int(os.getenv("APP_PORT", "8000"))
MODEL_CKPT = os.getenv("MODEL_CKPT", ""); N_VIEWS = int(os.getenv("N_VIEWS", "4"))
UNKNOWN_THR_BRAND = float(os.getenv("UNKNOWN_THR_BRAND", "0.55"))
UNKNOWN_THR_MODEL = float(os.getenv("UNKNOWN_THR_MODEL", "0.55"))
LABELS_DIR = Path(os.getenv("LABELS_DIR", str(BASE / "labels"))); DATASET_ROOT = os.getenv("DATASET_ROOT", "")

app = FastAPI(title="Golf Ball Sorter UI"); templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
server = Server()

def start_capture_server(): threading.Thread(target=accept_loop, args=(server,), daemon=True).start()
@app.on_event("startup")

def on_startup(): Path(SAVE_DIR).mkdir(parents=True, exist_ok=True); start_capture_server()
app.mount("/captures", StaticFiles(directory=str(SAVE_DIR)), name="captures")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
@app.get("/", response_class=HTMLResponse)

def index(request: Request):
    sessions = [p.name for p in Path(SAVE_DIR).glob("*") if p.is_dir()]; sessions.sort(reverse=True)
    with server.lock: cams = list(server.clients.keys())
    return templates.TemplateResponse("index.html", {"request": request, "sessions": sessions[:50], "cameras": cams, "model_ckpt_set": bool(MODEL_CKPT)})
@app.post("/capture")

def trigger_capture(background_tasks: BackgroundTasks):
    import uuid; sid = str(uuid.uuid4())[:8]; results = server.trigger_capture(sid)
    if MODEL_CKPT and results: background_tasks.add_task(run_inference_and_save, sid)
    return RedirectResponse(url=f"/session/{sid}", status_code=303)

from ballnet.models.heads import ordinal_logits_to_probs, brand_with_unknown
from ballnet.data.dataset import GRADE_ORDER
import torch

def infer_with_cached(session_dir: Path):
    # Load images & aux via ballnet.infer
    from ballnet.infer import load_folder
    views, aux, _ = load_folder(str(session_dir), N_VIEWS, 224)
    model = MODEL_CACHE["model"]
    brands = MODEL_CACHE["brands"]
    with torch.no_grad():
        out = model(views, aux)
        p_grade = ordinal_logits_to_probs(out["grade_logits"]).detach()[0].numpy()
        import torch.nn.functional as F
        brand_logits = out["brand_logits"]
        # Open-set routing
        other_idx = brand_logits.shape[1] - 1
        pred_idx, pred_conf = brand_with_unknown(brand_logits, UNKNOWN_THR_BRAND, other_idx)
        bi = int(pred_idx[0]); bconf = float(pred_conf[0])
        p_brand = F.softmax(brand_logits, dim=1)[0].numpy()
        att = out["attw"][0].numpy()
    gi = int(p_grade.argmax())
    return {
        "pred_grade": GRADE_ORDER[gi],
        "pred_grade_conf": float(p_grade[gi]),
        "grade_probs": {GRADE_ORDER[i]: float(p_grade[i]) for i in range(len(GRADE_ORDER))},
        "pred_brand": (brands[bi] if brands and 0 <= bi < len(brands) else str(bi)),
        "pred_brand_conf": bconf,
        "brand_probs": {brands[i]: float(p_brand[i]) for i in range(len(brands))},
        "att_weights": [float(x) for x in att.tolist()]
    }

def run_inference_and_save(session_id: str):

    try: outdir = Path(SAVE_DIR) / session_id; ensure_model_loaded(); pred = infer_with_cached(outdir) if MODEL_CACHE['model'] else infer_folder(str(outdir), MODEL_CKPT, views=N_VIEWS, img_size=224, unknown_thr_brand=UNKNOWN_THR_BRAND, unknown_thr_model=UNKNOWN_THR_MODEL); (outdir/"pred.json").write_text(json.dumps(pred, indent=2))
    except Exception as e: (Path(SAVE_DIR) / session_id / "pred.json").write_text(json.dumps({"error": str(e)}, indent=2))
@app.get("/session/{session_id}", response_class=HTMLResponse)

def session_view(request: Request, session_id: str):
    outdir = Path(SAVE_DIR) / session_id; images = sorted([p.name for p in outdir.glob("*.jpg")])
    pred_path = outdir / "pred.json"; pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    return templates.TemplateResponse("session.html", {"request": request, "session_id": session_id, "images": images, "pred": pred})

def append_label_csv(ball_id: str, image_name: str, grade: str, brand: str, model: str,
                     color: str, source: str, session_dir: Path, pred: dict | None):
    """Upsert a label row by ball_id into labels.csv (overwrite duplicates).

    Uniqueness is enforced by ball_id (session-level label). If a row for the
    same ball_id exists, it will be replaced. Writes atomically via a temp file.
    """
    import csv
    from datetime import datetime

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LABELS_DIR / "labels.csv"
    tmp_path = LABELS_DIR / "labels.tmp.csv"

    fields = [
        "ball_id","image_name","split","grade","brand","model","color",
        "source","session_path","brand_conf","grade_conf","model_conf","color_conf","timestamp",
    ]

    new_row = {
        "ball_id": ball_id,
        "image_name": image_name,
        "split": "train",
        "grade": grade,
        "brand": brand,
        "model": model,
        "color": color,
        "source": source,
        "session_path": str(session_dir),
        "brand_conf": (pred or {}).get("pred_brand_conf", ""),
        "grade_conf": (pred or {}).get("pred_grade_conf", ""),
        "model_conf": (pred or {}).get("pred_model_conf", ""),
        "color_conf": (pred or {}).get("pred_color_conf", ""),
        "timestamp": datetime.utcnow().isoformat(),
    }

    rows = []
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # keep all rows except those with the same ball_id
                    if (r.get("ball_id") or "") != ball_id:
                        rows.append({k: r.get(k, "") for k in fields})
        except Exception:
            # If reading fails (malformed file), fall back to recreating with the new row only
            rows = []

    rows.append(new_row)

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Atomic replace
    try:
        tmp_path.replace(csv_path)
    except Exception:
        # As a fallback on platforms that may not support replace semantics, copy contents
        with open(tmp_path, "r", encoding="utf-8") as src, open(csv_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())

def maybe_copy_to_dataset(ball_id: str, session_dir: Path):
    if not DATASET_ROOT: return
    dst = Path(DATASET_ROOT) / "images" / ball_id; dst.mkdir(parents=True, exist_ok=True)
    for p in session_dir.glob("*.jpg"): shutil.copy2(p, dst / p.name)
@app.post("/session/{session_id}/label")
def save_label(session_id: str, brand: str = Form(...), grade: str = Form(...),
               model: str = Form("Unknown"), color: str = Form("Unknown"),
               source: str = Form("human")):
    """Legacy form endpoint for saving labels."""
    outdir = Path(SAVE_DIR) / session_id
    pred_path = outdir / "pred.json"
    pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    append_label_csv(session_id, "ALL", grade, brand, model, color, source, outdir, pred)
    maybe_copy_to_dataset(session_id, outdir)
    LAST_LABEL.update({"brand": brand, "grade": grade, "color": color, "model": model})
    return RedirectResponse(url=f"/session/{session_id}", status_code=303)


@app.get("/api/session/{ball_id}")
def api_session(ball_id: str):
    """Return images and prediction for a session."""
    outdir = Path(SAVE_DIR) / ball_id
    if not outdir.exists():
        raise HTTPException(status_code=404, detail="session not found")
    images = sorted([p.name for p in outdir.glob("*.jpg")])
    pred_path = outdir / "pred.json"
    pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    return {"ball_id": ball_id, "images": images, "pred": pred}


@app.get("/api/last-label")
def api_last_label():
    """Return the last label saved (for sticky defaults)."""
    return LAST_LABEL


@app.get("/api/suggest")
def api_suggest(ball_id: str, image_name: str):
    """Return model suggestions if available; otherwise 204."""
    outdir = Path(SAVE_DIR) / ball_id
    pred_path = outdir / "pred.json"
    if not pred_path.exists():
        raise HTTPException(status_code=204)
    pred = json.loads(pred_path.read_text())
    return {"brand": pred.get("pred_brand"),
            "grade": pred.get("pred_grade"),
            "brand_conf": pred.get("pred_brand_conf"),
            "grade_conf": pred.get("pred_grade_conf")}


@app.get("/api/label-suggestions")
def api_label_suggestions(field: str, q: str = "", limit: int = 20):
    """Return unique label suggestions from labels.csv.
    field: one of 'brand','model','grade','color'
    q: optional case-insensitive prefix filter
    """
    field = field.lower()
    allowed = {"brand", "model", "grade", "color"}
    if field not in allowed:
        raise HTTPException(status_code=400, detail="invalid field")
    csv_path = LABELS_DIR / "labels.csv"
    if not csv_path.exists():
        return {"items": []}
    import csv
    values = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = (row.get(field) or "").strip()
                if not val:
                    continue
                values[val] = values.get(val, 0) + 1
    except Exception as e:
        # If CSV is malformed, fail softly
        return {"items": []}
    items = list(values.items())
    # Filter by prefix if q provided
    if q:
        ql = q.lower()
        items = [it for it in items if it[0].lower().startswith(ql)]
    # Sort by frequency desc, then alpha
    items.sort(key=lambda x: (-x[1], x[0].lower()))
    return {"items": [k for k, _ in items[:max(1, min(limit, 100))]]}


@app.get("/api/attn/{ball_id}/{image_name}")
def api_attention(ball_id: str, image_name: str, head: str = "grade"):
    """Serve a Grad-CAM attention overlay as a PNG image."""
    outdir = Path(SAVE_DIR) / ball_id
    img_path = outdir / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404)
    ensure_model_loaded()
    rgb = imread_rgb(str(img_path), 224)
    Lm, bm = estimate_ball_lab_features(rgb)
    views = to_tensor(rgb).unsqueeze(0)
    aux = torch.tensor([[Lm, bm]], dtype=torch.float32)
    if MODEL_CACHE["model"] is None:
        raise HTTPException(status_code=204)
    head = (head or "grade").lower()
    if head not in ("grade", "brand"):
        raise HTTPException(status_code=400, detail="invalid head")
    cam = gradcam_for_view(MODEL_CACHE["model"], views, aux, target_head=head,
                           n_views=N_VIEWS, device="cpu")
    overlay = overlay_heatmap(rgb.astype(numpy.uint8), cam, alpha=0.35)
    import cv2
    out_path = outdir / f"attn_{head}_{image_name}"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return FileResponse(out_path, media_type="image/png")


@app.post("/api/label")
def api_label(payload: dict = Body(...)):
    """Save a session-level label via JSON API."""
    ball_id = payload.get("ball_id")
    brand = payload.get("brand", "Unknown")
    grade = payload.get("grade", "Unknown")
    model = payload.get("model", "Unknown")
    color = payload.get("color", "Unknown")
    source = payload.get("source", "human")
    if not ball_id:
        raise HTTPException(status_code=400, detail="ball_id required")
    outdir = Path(SAVE_DIR) / ball_id
    pred_path = outdir / "pred.json"
    pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    append_label_csv(ball_id, "ALL", grade, brand, model, color, source, outdir, pred)
    maybe_copy_to_dataset(ball_id, outdir)
    LAST_LABEL.update({"brand": brand, "grade": grade, "color": color, "model": model})
    return {"status": "ok"}
@app.get("/session/{session_id}/gradcam/{image_name}", response_class=HTMLResponse)

def gradcam_view(request: Request, session_id: str, image_name: str, head: str = "grade"):
    outdir = Path(SAVE_DIR) / session_id; img_path = outdir / image_name
    pred = None; pred_path = outdir / "pred.json"
    if pred_path.exists():
        try: pred = json.loads(pred_path.read_text())
        except Exception: pred = None
    if not img_path.exists():
        return templates.TemplateResponse("session.html", {"request": request, "session_id": session_id, "images": [], "pred": pred})
    rgb = imread_rgb(str(img_path), 224); Lm,bm = estimate_ball_lab_features(rgb)
    views = to_tensor(rgb).unsqueeze(0); aux = torch.tensor([[Lm, bm]], dtype=torch.float32)
    if not MODEL_CKPT:
        images = sorted([p.name for p in outdir.glob('*.jpg')])
        return templates.TemplateResponse("session.html", {"request": request, "session_id": session_id, "images": images, "pred": pred})
    ensure_model_loaded(); model = MODEL_CACHE['model']; cam = gradcam_for_view(model, views, aux, target_head=head, n_views=N_VIEWS, device="cpu")
    overlay = overlay_heatmap(rgb.astype(numpy.uint8), cam, alpha=0.35)
    import cv2; out_name = f"gradcam_{head}_{image_name}"
    cv2.imwrite(str(outdir / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    images = sorted([p.name for p in outdir.glob('*.jpg')]); 
    if out_name not in images: images.append(out_name)
    return templates.TemplateResponse("session.html", {"request": request, "session_id": session_id, "images": images, "pred": pred})

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host=APP_HOST, port=APP_PORT)
