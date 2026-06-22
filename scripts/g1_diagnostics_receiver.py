#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small HTTP receiver for G1 diagnostics zip uploads."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def safe_filename(name: str) -> str:
    name = name.strip() or f"g1_diagnostics_{int(time.time())}.zip"
    name = name.split("/")[-1].split("\\")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.endswith(".zip"):
        name += ".zip"
    return name


def make_handler(out_dir: Path, unpack: bool):
    def safe_extract(zf: zipfile.ZipFile, dst: Path) -> None:
        dst_resolved = dst.resolve()
        for member in zf.infolist():
            target = (dst / member.filename).resolve()
            if os.path.commonpath([str(dst_resolved), str(target)]) != str(dst_resolved):
                raise ValueError(f"unsafe zip member path: {member.filename}")
        zf.extractall(dst)

    class UploadHandler(BaseHTTPRequestHandler):
        server_version = "G1DiagnosticsReceiver/1.0"

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._send_json(
                200,
                {
                    "ok": True,
                    "message": "POST diagnostics zip to /upload",
                    "out_dir": str(out_dir),
                    "unpack": unpack,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/upload":
                self._send_json(404, {"ok": False, "error": "use POST /upload"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._send_json(400, {"ok": False, "error": "empty body"})
                return

            query = parse_qs(parsed.query)
            filename = (
                query.get("filename", [""])[0]
                or self.headers.get("X-G1-Diagnostics-Filename", "")
                or f"g1_diagnostics_{int(time.time())}.zip"
            )
            filename = safe_filename(filename)
            out_dir.mkdir(parents=True, exist_ok=True)
            zip_path = out_dir / filename
            body = self.rfile.read(length)
            zip_path.write_bytes(body)

            unpack_dir = None
            if unpack:
                unpack_dir = out_dir / filename[:-4]
                unpack_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    safe_extract(zf, unpack_dir)

            payload = {
                "ok": True,
                "saved_zip": str(zip_path),
                "num_bytes": len(body),
                "unpacked_to": str(unpack_dir) if unpack_dir else None,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            self._send_json(200, payload)

    return UploadHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive G1 diagnostics zip uploads.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out-dir", default="./g1_diagnostics_uploads")
    parser.add_argument("--unpack", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    handler = make_handler(out_dir, args.unpack)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Listening on http://{args.host}:{args.port}/upload")
    print(f"Saving uploads to {out_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping receiver.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
