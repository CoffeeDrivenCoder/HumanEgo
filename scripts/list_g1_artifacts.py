#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""List G1/HumanEgo artifact sessions and runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from g1_artifacts import artifact_root, artifact_session, artifact_dir


def dir_info(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    return {
        "path": str(path),
        "exists": path.exists(),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def list_runs(role_dir: Path) -> list[dict]:
    if not role_dir.exists():
        return []
    runs = []
    for path in sorted(p for p in role_dir.iterdir() if p.is_dir()):
        info = dir_info(path)
        info["name"] = path.name
        info["mtime"] = path.stat().st_mtime
        runs.append(info)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=artifact_session())
    parser.add_argument("--root", type=Path, default=artifact_root())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    session_dir = args.root.expanduser().resolve() / args.session
    payload = {
        "root": str(args.root.expanduser().resolve()),
        "session": args.session,
        "session_dir": str(session_dir),
        "roles": {},
    }
    for role in ("server", "client", "interactive", "diagnostics"):
        role_dir = session_dir / role
        runs = list_runs(role_dir)
        payload["roles"][role] = {
            "path": str(role_dir),
            "runs": runs,
            "latest": runs[-1]["path"] if runs else None,
        }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"artifact root: {payload['root']}")
    print(f"session: {payload['session']}")
    for role, info in payload["roles"].items():
        runs = info["runs"]
        print(f"\n{role}: {info['path']}")
        if not runs:
            print("  no runs")
            continue
        for run in runs[-10:]:
            mib = run["bytes"] / (1024 * 1024)
            print(f"  {run['name']}  files={run['files']}  size={mib:.2f} MiB")
        print(f"  latest: {info['latest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
