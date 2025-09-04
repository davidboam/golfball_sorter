#!/usr/bin/env python3
import argparse, subprocess, os, sys
from pathlib import Path

CAPTURE_SERVER = Path(__file__).parent / "server.py"
INFER_SCRIPT   = Path(__file__).parents[1] / "ml" / "infer_ballnet.py"

def run_server():
    print("Starting server... (Ctrl+C to stop)")
    env = os.environ.copy()
    subprocess.run([sys.executable, str(CAPTURE_SERVER)], env=env, check=False)

def capture_once_and_infer(model_ckpt: str, n_views: int = 4):
    cap_root = Path(os.getenv("SAVE_DIR", str(Path(__file__).resolve().parents[1] / "captures")))
    before = set([p.name for p in cap_root.glob("*") if p.is_dir()])
    input("Trigger a capture in the server window now. Press Enter here when saved...")
    after = set([p.name for p in cap_root.glob("*") if p.is_dir()])
    new = sorted(list(after - before))
    if not new:
        print("No new capture detected. Did you trigger?")
        return
    cap_dir = cap_root / new[-1]
    print("New capture:", cap_dir)
    cmd = [sys.executable, str(INFER_SCRIPT), "--images", str(cap_dir), "--ckpt", model_ckpt, "--n-views", str(n_views)]
    subprocess.run(cmd, check=False)

def main():
    ap = argparse.ArgumentParser(description="Capture/Inference CLI")
    ap.add_argument("--serve", action="store_true", help="Run capture server")
    ap.add_argument("--infer-once", type=str, default=None, help="Run inference on the latest capture using CKPT path")
    ap.add_argument("--n-views", type=int, default=4)
    args = ap.parse_args()

    if args.serve:
        run_server()
    elif args.infer_once:
        capture_once_and_infer(args.infer_once, n_views=args.n_views)
    else:
        print("Nothing to do. Use --serve or --infer-once <ckpt>")

if __name__ == "__main__":
    main()
