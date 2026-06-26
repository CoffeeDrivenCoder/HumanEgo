#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared artifact paths for G1/HumanEgo validation scripts."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "g1_humanego"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def default_session() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def safe_name(value: str) -> str:
    value = str(value).strip()
    if not value:
        return "untagged"
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "untagged"


def artifact_root() -> Path:
    return Path(os.getenv("G1_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))).expanduser().resolve()


def artifact_session() -> str:
    return safe_name(os.getenv("G1_ARTIFACT_SESSION", default_session()))


def artifact_dir(*parts: str | os.PathLike[str]) -> Path:
    root = artifact_root() / artifact_session()
    for part in parts:
        text = str(part)
        if text:
            root = root / safe_name(text)
    return root


def run_dir(role: str, tag: str, prefix: str | None = None, stamp: str | None = None) -> Path:
    stamp = stamp or utc_stamp()
    role = safe_name(role)
    tag = safe_name(tag)
    if prefix:
        name = f"{safe_name(prefix)}_{stamp}_{tag}"
    else:
        name = f"{stamp}_{tag}"
    return artifact_dir(role, name)


def legacy_dir(name: str) -> Path:
    return PROJECT_ROOT / name
