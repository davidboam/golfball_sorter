#!/usr/bin/env python3
import os, json, threading, shutil
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sys
import numpy

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE / "capture")); from server import Server, accept_loop, SAVE_DIR
sys.path.append(str(BASE / "ml")); from infer_ballnet import infer_folder

from ml.train_ballnet import imread_rgb, to_tensor, estimate_ball_lab_features, MultiViewBallNet, GRADE_ORDER
from utils.gradcam import gradcam_for_view, overlay_heatmap
import torch
# ---- Model cache ----
MODEL_CACHE = {"model": None, "brands": None, "args": None}

def load_model_from_ckpt(ckpt_path: str):
    if not ckpt_path:
        return None
    import torch
    from ml.train_ballnet import MultiViewBallNet, GRADE_ORDER
    ckpt = torch.load(ckpt_path, map_location="cpu")
    brands = ckpt.get("brands", ckpt.get("brand_list", []))
    backbone = ckpt["args"].get("backbone","mobilenetv3_small_100")
    emb_dim = ckpt["args"].get("emb_dim",256)
    model = MultiViewBallNet(backbone_name=backbone, n_grades=len(GRADE_ORDER),
                             n_brands=len(brands), emb_dim=emb_dim, aux_dim=2)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, brands, ckpt["args"]

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

from ml.train_ballnet import ordinal_logits_to_probs, GRADE_ORDER
import torch

def infer_with_cached(session_dir: Path):
    # Load images & aux like ml/infer_ballnet
    from ml.infer_ballnet import load_images
    views, aux = load_images(session_dir, N_VIEWS)
    model = MODEL_CACHE["model"]
    brands = MODEL_CACHE["brands"]
    with torch.no_grad():
        glog, blog, attw = model(views, aux)
        p_grade = ordinal_logits_to_probs(glog)[0].numpy()
        import torch.nn.functional as F
        p_brand = F.softmax(blog, dim=1)[0].numpy()
        att = attw[0].numpy()
    gi = int(p_grade.argmax()); bi = int(p_brand.argmax())
    return {
        "pred_grade": GRADE_ORDER[gi],
        "pred_grade_conf": float(p_grade[gi]),
        "grade_probs": {GRADE_ORDER[i]: float(p_grade[i]) for i in range(len(GRADE_ORDER))},
        "pred_brand": brands[bi],
        "pred_brand_conf": float(p_brand[bi]),
        "brand_probs": {brands[i]: float(p_brand[i]) for i in range(len(brands))},
        "att_weights": [float(x) for x in att.tolist()]
    }

def run_inference_and_save(session_id: str):

    try: outdir = Path(SAVE_DIR) / session_id; ensure_model_loaded(); pred = infer_with_cached(outdir) if MODEL_CACHE['model'] else infer_folder(str(outdir), MODEL_CKPT, n_views=N_VIEWS); (outdir/"pred.json").write_text(json.dumps(pred, indent=2))
    except Exception as e: (Path(SAVE_DIR) / session_id / "pred.json").write_text(json.dumps({"error": str(e)}, indent=2))
@app.get("/session/{session_id}", response_class=HTMLResponse)

def session_view(request: Request, session_id: str):
    outdir = Path(SAVE_DIR) / session_id; images = sorted([p.name for p in outdir.glob("*.jpg")])
    pred_path = outdir / "pred.json"; pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    return templates.TemplateResponse("session.html", {"request": request, "session_id": session_id, "images": images, "pred": pred})

def append_label_csv(ball_id: str, grade: str, brand: str, model: str, source: str, session_dir: Path, pred: dict | None):
    LABELS_DIR.mkdir(parents=True, exist_ok=True); csv_path = LABELS_DIR / "labels.csv"
    header = "ball_id,split,grade,brand,model,source,session_path,brand_conf,grade_conf,timestamp\n"
    line = f"{ball_id},train,{grade},{brand},{model},{source},{session_dir},{(pred or {}).get('pred_brand_conf','')},{(pred or {}).get('pred_grade_conf','')},{__import__('datetime').datetime.utcnow().isoformat()}\n"
    if not csv_path.exists(): csv_path.write_text(header + line)
    else:
        with open(csv_path, "a", encoding="utf-8") as f: f.write(line)

def maybe_copy_to_dataset(ball_id: str, session_dir: Path):
    if not DATASET_ROOT: return
    dst = Path(DATASET_ROOT) / "images" / ball_id; dst.mkdir(parents=True, exist_ok=True)
    for p in session_dir.glob("*.jpg"): shutil.copy2(p, dst / p.name)
@app.post("/session/{session_id}/label")

def save_label(session_id: str, brand: str = Form(...), grade: str = Form(...), model: str = Form("Unknown"), source: str = Form("human")):
    outdir = Path(SAVE_DIR) / session_id; pred_path = outdir / "pred.json"; pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
    append_label_csv(session_id, grade, brand, model, source, outdir, pred); maybe_copy_to_dataset(session_id, outdir)
    return RedirectResponse(url=f"/session/{session_id}", status_code=303)
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