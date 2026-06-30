#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locate RobotBundleData or equivalent bundle classes in robot-side packages."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import re
import sys
import traceback
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


DEFAULT_PACKAGES = ["a2d_sdk", "corobot", "genie_msgs_pb"]
DEFAULT_SYMBOL_PATTERNS = ["RobotBundleData", "BundleData", "Bundle", "RobotBundle"]


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


def inspect_module(name: str, symbol_patterns: list[str]) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "ok": False}
    try:
        spec = importlib.util.find_spec(name)
        item["spec_found"] = spec is not None
        item["origin"] = None if spec is None else spec.origin
    except Exception as exc:
        item["spec_error"] = f"{type(exc).__name__}: {exc}"
    try:
        module = importlib.import_module(name)
        item["ok"] = True
        item["file"] = getattr(module, "__file__", None)
        matches = {}
        for attr in dir(module):
            if any(pattern.lower() in attr.lower() for pattern in symbol_patterns):
                value = getattr(module, attr)
                entry: dict[str, Any] = {
                    "repr": repr(value),
                    "type": f"{type(value).__module__}.{type(value).__name__}",
                }
                if callable(value):
                    try:
                        entry["signature"] = str(inspect.signature(value))
                    except Exception as exc:
                        entry["signature_error"] = f"{type(exc).__name__}: {exc}"
                matches[attr] = entry
        item["symbol_matches"] = matches
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


def walk_package_modules(package_name: str, max_modules: int) -> list[str]:
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return []
    paths = getattr(package, "__path__", None)
    if not paths:
        return [package_name]
    names = [package_name]
    for idx, mod in enumerate(pkgutil.walk_packages(paths, prefix=f"{package_name}.")):
        if idx >= max_modules:
            break
        names.append(mod.name)
    return names


def grep_package_files(package_name: str, symbol_patterns: list[str], max_files: int) -> list[dict[str, Any]]:
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return []
    roots = []
    if getattr(package, "__file__", None):
        roots.append(Path(package.__file__).resolve().parent)
    for raw in getattr(package, "__path__", []) or []:
        roots.append(Path(raw).resolve())
    seen_roots = []
    for root in roots:
        if root not in seen_roots and root.exists():
            seen_roots.append(root)
    pattern_re = re.compile("|".join(re.escape(p) for p in symbol_patterns), re.IGNORECASE)
    out: list[dict[str, Any]] = []
    scanned = 0
    for root in seen_roots:
        for path in root.rglob("*"):
            if scanned >= max_files:
                return out
            if not path.is_file() or path.suffix not in {".py", ".pyi", ".proto", ".json", ".yaml", ".yml"}:
                continue
            scanned += 1
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            matches = []
            for lineno, line in enumerate(lines, start=1):
                if pattern_re.search(line):
                    matches.append({"line": lineno, "text": line[:500]})
            if matches:
                out.append({"path": str(path), "matches": matches[:80]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(artifact_dir("diagnostics")))
    parser.add_argument("--tag", default="robot_bundle_data_probe")
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--symbol-pattern", action="append", default=[])
    parser.add_argument("--max-modules-per-package", type=int, default=500)
    parser.add_argument("--max-files-per-package", type=int, default=3000)
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--upload-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    packages = DEFAULT_PACKAGES + list(args.package)
    symbol_patterns = DEFAULT_SYMBOL_PATTERNS + list(args.symbol_pattern)
    out_base = Path(args.out_dir).expanduser().resolve()
    default_base = artifact_dir("diagnostics")
    if out_base == default_base:
        run_dir = artifact_run_dir("diagnostics", args.tag, prefix="robot_bundle_data_probe")
    else:
        run_dir = out_base / f"robot_bundle_data_probe_{utc_stamp()}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    module_names: list[str] = []
    for package in packages:
        for name in walk_package_modules(package, args.max_modules_per_package):
            if name not in module_names:
                module_names.append(name)
    module_results = [inspect_module(name, symbol_patterns) for name in module_names]
    file_matches = {
        package: grep_package_files(package, symbol_patterns, args.max_files_per_package)
        for package in packages
    }

    report = {
        "ok": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "python": sys.executable,
        "packages": packages,
        "symbol_patterns": symbol_patterns,
        "module_results_with_symbol_matches": [
            item for item in module_results if item.get("symbol_matches")
        ],
        "module_errors": [
            item for item in module_results if not item.get("ok") and item.get("error")
        ],
        "num_modules_scanned": len(module_results),
        "file_matches": file_matches,
    }
    report_path = run_dir / "robot_bundle_data_probe_report.json"
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

    print(json.dumps({"run_dir": str(run_dir), "zip_path": str(zip_path), "report_path": str(report_path), "upload": upload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
