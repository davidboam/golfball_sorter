#!/usr/bin/env python3
import os, socket, struct, json, time
from io import BytesIO

from picamera2 import Picamera2
from libcamera import controls
from PIL import Image

SERVER_HOST = os.getenv("SERVER_HOST", "192.168.0.5")
SERVER_PORT = int(os.getenv("SERVER_PORT", "5001"))
CAMERA_ID   = os.getenv("CAMERA_ID", "pi-01")

def send_msg(sock, header: dict, payload: bytes = b""):
    h = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">I", len(h)) + h + payload)

def recv_msg(sock):
    raw = sock.recv(4)
    if not raw:
        return None, None
    (hlen,) = struct.unpack(">I", raw)
    hjson = b""
    while len(hjson) < hlen:
        chunk = sock.recv(hlen - len(hjson))
        if not chunk:
            raise ConnectionError("Socket closed while reading header")
        hjson += chunk
    header = json.loads(hjson.decode("utf-8"))
    payload = b""
    plen = header.get("payload_len", 0)
    while len(payload) < plen:
        chunk = sock.recv(plen - len(payload))
        if not chunk:
            raise ConnectionError("Socket closed while reading payload")
        payload += chunk
    return header, payload

def jpeg_bytes_from_array(arr, quality=90):
    im = Image.fromarray(arr)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def init_camera():
    picam = Picamera2()
    config = picam.create_still_configuration(buffer_count=2,main={"size": (1024,1024)})
    picam.configure(config)
    picam.start()
    time.sleep(0.5)
    picam.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": 1,
        # "ExposureTime": int(os.getenv("EXPOSURE_US", "10000")),
        # "AnalogueGain": float(os.getenv("ANALOG_GAIN", "1.8")),
        # "ColourGains": (float(os.getenv("CG_R", "1.8")), float(os.getenv("CG_B", "1.8")))
    })
    time.sleep(0.2)
    return picam

def capture_jpeg(picam):
    arr = picam.capture_array()
    return jpeg_bytes_from_array(arr, quality=int(os.getenv("JPEG_QUALITY", "90"))), arr.shape[1], arr.shape[0]

def main():
    picam = init_camera()
    while True:
        try:
            print(f"[{CAMERA_ID}] Connecting to {SERVER_HOST}:{SERVER_PORT} ...")
            with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=10) as s:
                send_msg(s, {"type": "hello", "camera_id": CAMERA_ID, "version": 1})
                while True:
                    header, _ = recv_msg(s)
                    if header is None:
                        break
                    if header.get("type") == "capture":
                        session_id = header.get("session_id")
                        ts = time.time()
                        jpg, w, h = capture_jpeg(picam)
                        resp = {
                            "type": "image",
                            "camera_id": CAMERA_ID,
                            "session_id": session_id,
                            "ts": ts,
                            "format": "jpg",
                            "width": w,
                            "height": h,
                            "payload_len": len(jpg),
                        }
                        send_msg(s, resp, jpg)
        except Exception as e:
            print(f"[{CAMERA_ID}] Disconnected/error: {e}. Reconnecting in 2s...")
            time.sleep(2)

if __name__ == "__main__":
    main()
