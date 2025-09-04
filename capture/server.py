#!/usr/bin/env python3
import os, socket, struct, json, threading, time, uuid
from pathlib import Path

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5001"))
SAVE_DIR = Path(os.getenv("SAVE_DIR", str(Path(__file__).resolve().parents[1] / "captures")))
TIMEOUT_S = float(os.getenv("TIMEOUT_S", "5.0"))

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

class Client(threading.Thread):
    def __init__(self, sock, addr, server):
        super().__init__(daemon=True)
        self.sock, self.addr, self.server = sock, addr, server
        self.camera_id = None
        self.alive = True

    def run(self):
        try:
            header, _ = recv_msg(self.sock)
            if not header or header.get("type") != "hello":
                raise ConnectionError("Did not receive hello")
            self.camera_id = header.get("camera_id", f"{self.addr[0]}:{self.addr[1]}")
            self.server.register(self.camera_id, self)
            print(f"[+] {self.camera_id} connected from {self.addr}")
            while self.alive:
                header, payload = recv_msg(self.sock)
                if header is None:
                    break
                t = header.get("type")
                if t == "image":
                    self.server.handle_image(header, payload)
                elif t == "pong":
                    pass
        except Exception as e:
            print(f"[!] {self.camera_id or self.addr} error: {e}")
        finally:
            self.alive = False
            self.server.unregister(self.camera_id, self)
            try: self.sock.close()
            except: pass
            print(f"[-] {self.camera_id or self.addr} disconnected")

    def capture(self, session_id):
        send_msg(self.sock, {"type": "capture", "session_id": session_id})

class Server:
    def __init__(self):
        self.clients = {}     # camera_id -> Client
        self.lock = threading.Lock()
        self.pending = {}     # session_id -> dict(camera_id -> saved_path)

    def register(self, camera_id, client):
        with self.lock:
            self.clients[camera_id] = client

    def unregister(self, camera_id, client):
        with self.lock:
            if camera_id in self.clients and self.clients[camera_id] is client:
                del self.clients[camera_id]

    def handle_image(self, header, payload):
        session_id = header["session_id"]
        camera_id  = header["camera_id"]
        outdir = SAVE_DIR / session_id
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / f"{camera_id}.jpg"
        with open(outpath, "wb") as f:
            f.write(payload)
        with self.lock:
            self.pending.setdefault(session_id, {})[camera_id] = str(outpath)
        print(f"    saved {camera_id} -> {outpath}")

    def trigger_capture(self, session_id, wait_timeout=TIMEOUT_S):
        with self.lock:
            clients = list(self.clients.items())
            self.pending[session_id] = {}
        if not clients:
            print("No clients connected.")
            return {}
        print(f"Triggering capture for {len(clients)} cams, session {session_id} ...")
        for cam_id, client in clients:
            try:
                client.capture(session_id)
            except Exception as e:
                print(f"Failed to trigger {cam_id}: {e}")

        t0 = time.time()
        while time.time() - t0 < wait_timeout:
            with self.lock:
                got = len(self.pending.get(session_id, {}))
            if got >= len(clients):
                break
            time.sleep(0.05)

        with self.lock:
            results = dict(self.pending.get(session_id, {}))
        missing = set(dict(clients).keys()) - set(results.keys())
        if missing:
            print(f"Timeout waiting for: {sorted(missing)}")
        else:
            print("All images received.")
        return results

def accept_loop(server):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            Client(conn, addr, server).start()

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    srv = Server()
    threading.Thread(target=accept_loop, args=(srv,), daemon=True).start()
    print("Commands: [enter]=capture, 'q'=quit, 'ls'=list cameras")
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            cmd = "q"
        if cmd == "q":
            break
        elif cmd == "ls":
            with srv.lock:
                print("Connected cams:", list(srv.clients.keys()))
        else:
            sid = str(uuid.uuid4())[:8]
            results = srv.trigger_capture(sid)
            print("Captured:", results)

if __name__ == "__main__":
    main()
