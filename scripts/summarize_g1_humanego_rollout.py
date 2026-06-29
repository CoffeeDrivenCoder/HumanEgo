#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize a G1 HumanEgo autoregressive rollout JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_rollout(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "autoregressive_rollout" in data:
        return data["autoregressive_rollout"]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_json")
    args = parser.parse_args()
    rollout = load_rollout(Path(args.rollout_json).expanduser().resolve())
    steps = rollout.get("steps") or []
    print(f"rollout: steps={len(steps)} target_source={rollout.get('target_source')} update_gripper={rollout.get('update_gripper')}")
    print("idx  trans_cm  rot_deg  obj1_dcm  closer  gripper  raw_cm  raw_rot  clipped")
    trans = []
    rots = []
    obj1_deltas = []
    obj1_closer = 0
    for item in steps:
        selected = item.get("selected") or {}
        raw = item.get("raw") or {}
        limited = item.get("limited") or {}
        obj1 = ((item.get("approach_metrics") or {}).get("obj1") or {})
        lim_info = limited.get("safety_translation_limit") or {}
        trans_m = float(selected.get("delta_norm_m") or 0.0)
        rot_deg = float(selected.get("rotation_delta_deg") or 0.0)
        raw_m = float(raw.get("delta_norm_m") or 0.0)
        raw_rot = float(raw.get("rotation_delta_deg") or 0.0)
        obj1_delta_m = obj1.get("target_minus_current_m")
        obj1_delta_cm = None if obj1_delta_m is None else float(obj1_delta_m) * 100.0
        closer = bool(obj1.get("closer", False))
        trans.append(trans_m)
        rots.append(rot_deg)
        if obj1_delta_cm is not None:
            obj1_deltas.append(obj1_delta_cm)
            obj1_closer += int(closer)
        print(
            f"{int(item.get('idx', -1)):02d} "
            f"{trans_m * 100:8.2f} "
            f"{rot_deg:8.2f} "
            f"{obj1_delta_cm if obj1_delta_cm is not None else float('nan'):9.2f} "
            f"{str(closer):>6s} "
            f"{float(item.get('gripper_target_0_open_1_closed') or 0.0):8.3f} "
            f"{raw_m * 100:7.2f} "
            f"{raw_rot:7.2f} "
            f"{lim_info.get('clipped')}"
        )
    if steps:
        print()
        print(
            "translation cm avg/min/max:",
            f"{np.mean(trans) * 100:.2f}",
            f"{np.min(trans) * 100:.2f}",
            f"{np.max(trans) * 100:.2f}",
        )
        print(
            "rotation deg avg/min/max:",
            f"{np.mean(rots):.2f}",
            f"{np.min(rots):.2f}",
            f"{np.max(rots):.2f}",
        )
        print("large steps trans>3cm:", sum(v > 0.03 for v in trans))
        print("large rotations rot>10deg:", sum(v > 10.0 for v in rots))
        if obj1_deltas:
            print(
                "obj1 target-current distance delta cm avg/min/max:",
                f"{np.mean(obj1_deltas):.2f}",
                f"{np.min(obj1_deltas):.2f}",
                f"{np.max(obj1_deltas):.2f}",
            )
            print("obj1 closer steps:", f"{obj1_closer}/{len(obj1_deltas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
