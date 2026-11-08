#!/usr/bin/env python3
"""MJPEG debug view: open http://<pi-ip>:8080 in a browser.

Standalone:   python tools/stream.py --profile profiles/default.yaml
From run.py:  python run.py ... --stream   (uses MjpegServer below)

Standalone mode shows the camera with the vision overlay; motors are never touched.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

_PAGE = b"""<!doctype html><html><head><title>PiRacer</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0;text-align:center}
img{width:100%;max-width:960px;image-rendering:pixelated}</style></head>
<body><img src="/stream"></body></html>"""


class MjpegServer:
    def __init__(self, port: int = 8080, quality: int = 70, max_fps: float = 15.0):
        self.quality = quality
        self.min_dt = 1.0 / max_fps
        self._jpeg = b""
        self._last = 0.0
        self._cond = threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_PAGE)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with server._cond:
                            server._cond.wait(timeout=1.0)
                            jpg = server._jpeg
                        if not jpg:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True, name="mjpeg").start()

    def publish(self, bgr) -> None:
        """Rate-limited; cheap to call every loop."""
        now = time.monotonic()
        if now - self._last < self.min_dt:
            return
        self._last = now
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._cond:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()

    def close(self) -> None:
        self.httpd.shutdown()


def main():
    from racer import profile
    from racer.camera import PiCamera
    from racer.vision import Vision, draw_debug

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="profiles/default.yaml")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--auto-exposure", action="store_true", help="ignore locked camera values")
    ap.add_argument("--raw", action="store_true", help="no vision overlay")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    prof = profile.load(args.profile, args.set)
    cam = PiCamera(prof.camera, auto=args.auto_exposure)
    vision = Vision(prof.vision, prof.camera.width, prof.camera.height)
    srv = MjpegServer(args.port)
    print(f"streaming on http://0.0.0.0:{args.port}  (Ctrl-C to stop)")
    idx = -1
    try:
        while True:
            f = cam.get(idx, timeout=0.5)
            if f is None:
                continue
            img, _, idx = f
            if args.raw:
                srv.publish(img)
                continue
            est = vision.process(img, keep_mask=True)
            srv.publish(draw_debug(img, vision, est, text=f"valid={est.valid}"))
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
