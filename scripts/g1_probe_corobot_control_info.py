#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only probe for robot-side CoRobot control capabilities.

The script does not send motion commands. It inspects local Python modules,
common CoRobot package paths, and read-only/OPTIONS HTTP responses so we can
decide whether HumanEgo can call RoboClaw-style EEF_ABS actions on this robot.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import platform
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from g1_artifacts import artifact_dir, run_dir as artifact_run_dir  # noqa: E402
from g1_humanego_client_dry_run import json_safe, upload_zip  # noqa: E402


DEFAULT_COROBOT_BASE_URL = "http://localhost:8765"
DEFAULT_ENDPOINTS = [
    "/status",
    "/get_prompt",
    "/skill/move_eef",
    "/skill/execute_action",
    "/execute_action",
    "/action",
    "/system/start_policytask",
    "/system/stop_policytask",
    "/system/reset_policytask",
    "/set_evaluate_params",
]
DEFAULT_MODULES = [
    "corobot",
    "corobot.utils.kinematics",
    "corobot.utils.fk_solver",
    "corobot.policy_tasks.rule_control_task",
    "a2d_sdk.robot",
]
INTERESTING_PATTERNS = [
    "EEF_ABS",
    "move_eef",
    "execute_action",
    "trajectory_reference_time",
    "kind",
    "_execute",
    "ABS_JOINT",
    "DELTA_POSE",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_zip(src_dir: Path) -> Path:
    zip_path = src_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))
    return zip_path


def read_text_head(path: Path, max_bytes: int = 200_000) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def safe_import_module(name: str) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "ok": False}
    try:
        spec = importlib.util.find_spec(name)
        item["spec_found"] = spec is not None
        item["origin"] = None if spec is None else spec.origin
        item["submodule_search_locations"] = (
            None if spec is None or spec.submodule_search_locations is None else list(spec.submodule_search_locations)
        )
    except Exception as exc:
        item.update({"spec_error_type": type(exc).__name__, "spec_error": str(exc)})
    try:
        module = importlib.import_module(name)
        item["ok"] = True
        item["file"] = getattr(module, "__file__", None)
        item["package"] = getattr(module, "__package__", None)
        exported = [key for key in dir(module) if not key.startswith("__")]
        item["exported_head"] = exported[:80]
        interesting = {}
        for key in exported:
            if any(pattern.lower() in key.lower() for pattern in ("eef", "action", "execute", "control", "task")):
                value = getattr(module, key, None)
                entry = {"repr": repr(value)}
                if callable(value):
                    try:
                        entry["signature"] = str(inspect.signature(value))
                    except Exception as exc:
                        entry["signature_error"] = f"{type(exc).__name__}: {exc}"
                interesting[key] = entry
        item["interesting_exports"] = interesting
    except Exception as exc:
        item.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return item


def probe_robot_controller() -> dict[str, Any]:
    item: dict[str, Any] = {"ok": False}
    try:
        from a2d_sdk.robot import RobotController

        controller = RobotController()
        item["ok"] = True
        item["class"] = f"{type(controller).__module__}.{type(controller).__name__}"
        methods = {}
        for name in dir(controller):
            if name.startswith("_"):
                continue
            if any(token in name.lower() for token in ("control", "action", "eef", "pose", "trajectory", "move", "execute")):
                value = getattr(controller, name)
                entry = {"repr": repr(value), "callable": callable(value)}
                if callable(value):
                    try:
                        entry["signature"] = str(inspect.signature(value))
                    except Exception as exc:
                        entry["signature_error"] = f"{type(exc).__name__}: {exc}"
                methods[name] = entry
        item["methods"] = methods
    except Exception as exc:
        item.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return item


def http_request(method: str, url: str, timeout_s: float) -> dict[str, Any]:
    started = time.time()
    req = urllib.request.Request(url, method=method, headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(20_000).decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(body)
            except Exception:
                parsed = body
            return {
                "ok": 200 <= int(resp.status) < 300,
                "method": method,
                "url": url,
                "status": int(resp.status),
                "headers": dict(resp.headers.items()),
                "duration_s": time.time() - started,
                "body": parsed,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(20_000).decode("utf-8", errors="replace")
        return {
            "ok": False,
            "method": method,
            "url": url,
            "status": int(exc.code),
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "duration_s": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "body": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "method": method,
            "url": url,
            "duration_s": time.time() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def probe_http(base_url: str, endpoints: list[str], timeout_s: float) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    out: dict[str, Any] = {"base_url": base_url, "endpoints": {}}
    for endpoint in endpoints:
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{base_url}{endpoint}"
        methods = ["OPTIONS"]
        if endpoint in {"/status", "/get_prompt"}:
            methods.insert(0, "GET")
        endpoint_out = {}
        for method in methods:
            endpoint_out[method] = http_request(method, url, timeout_s)
        out["endpoints"][endpoint] = endpoint_out
    return out


def candidate_corobot_roots(extra_roots: list[str]) -> list[Path]:
    roots = [
        "~/.a2d_pkg/corobot",
        "~/.a2d_pkg",
        "/home/ke/.a2d_pkg/corobot",
        "/home/ke/.a2d_pkg",
        "/home/agiuser/.a2d_pkg/corobot",
        "/home/agiuser/.a2d_pkg",
        "/home/ubuntu/.a2d_pkg/corobot",
        "/home/ubuntu/.a2d_pkg",
        "/opt/corobot",
        "/usr/local/lib/python3.10/site-packages/corobot",
        "/home/ke/miniconda3/envs/a2d/lib/python3.10/site-packages/corobot",
    ]
    roots.extend(extra_roots)
    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in roots:
        path = Path(raw).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            resolved.append(path)
    return resolved


def grep_file(path: Path) -> dict[str, Any] | None:
    try:
        text = read_text_head(path)
    except Exception:
        return None
    matches: list[dict[str, Any]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        if any(pattern in line for pattern in INTERESTING_PATTERNS):
            matches.append({"line": idx, "text": line[:500]})
    if not matches:
        return None
    funcs: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("@expose_api"):
            funcs.append({"line": idx, "text": stripped[:500]})
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "matches": matches[:120],
        "definitions": funcs[:120],
    }


def scan_paths(roots: list[Path], max_files: int) -> dict[str, Any]:
    out: dict[str, Any] = {"roots": [], "files_with_matches": []}
    count = 0
    for root in roots:
        root_item: dict[str, Any] = {"path": str(root), "exists": root.exists(), "is_dir": root.is_dir()}
        out["roots"].append(root_item)
        if not root.exists():
            continue
        files = []
        if root.is_file():
            files = [root]
        else:
            try:
                files = [p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".yml", ".yaml", ".json", ".toml"}]
            except Exception as exc:
                root_item["scan_error"] = f"{type(exc).__name__}: {exc}"
                continue
        root_item["num_candidate_files"] = len(files)
        for path in files:
            if count >= max_files:
                out["truncated"] = True
                return out
            count += 1
            match = grep_file(path)
            if match:
                out["files_with_matches"].append(match)
    out["num_scanned_files"] = count
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="corobot_control_probe")
    parser.add_argument("--corobot-base-url", default=os.getenv("COROBOT_BASE_URL", DEFAULT_COROBOT_BASE_URL))
    parser.add_argument("--http-timeout-s", type=float, default=2.0)
    parser.add_argument("--extra-endpoint", action="append", default=[])
    parser.add_argument("--extra-module", action="append", default=[])
    parser.add_argument("--extra-root", action="append", default=[])
    parser.add_argument("--max-scan-files", type=int, default=5000)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="corobot_control_probe")
    else:
        run_dir = out_base / f"corobot_control_probe_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    modules = DEFAULT_MODULES + list(args.extra_module)
    endpoints = DEFAULT_ENDPOINTS + list(args.extra_endpoint)
    roots = candidate_corobot_roots(args.extra_root)

    report: dict[str, Any] = {
        "ok": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "platform": {
            "python": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "env_path": os.environ.get("PATH"),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "notes": [
            "Read-only probe: no robot motion commands are sent.",
            "HTTP probe uses GET only for /status and /get_prompt, OPTIONS for candidate control endpoints.",
        ],
        "module_probe": [safe_import_module(name) for name in modules],
        "robot_controller_probe": probe_robot_controller(),
        "http_probe": probe_http(args.corobot_base_url, endpoints, args.http_timeout_s),
        "path_scan": scan_paths(roots, args.max_scan_files),
    }

    report_path = run_dir / "corobot_control_probe_report.json"
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = make_zip(run_dir)
    upload = None
    if args.upload_url:
        try:
            upload = upload_zip(zip_path, args.upload_url, args.upload_timeout_s)
        except Exception as exc:
            upload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (run_dir / "upload_result.json").write_text(json.dumps(json_safe(upload), ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = make_zip(run_dir)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "zip_path": str(zip_path),
                "report_path": str(report_path),
                "upload": upload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
