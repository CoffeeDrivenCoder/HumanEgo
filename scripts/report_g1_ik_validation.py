#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print compact summaries for G1 FK/IK validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def print_offline(data: dict[str, Any]) -> None:
    sides = data.get("sides") or {}
    for side, rep in sides.items():
        summary = rep.get("summary") or {}
        current = ((rep.get("current_pose_self_consistency") or {}).get("ik") or {})
        print(
            f"{side}: current_success={summary.get('current_success')} "
            f"small={summary.get('small_successes')}/{summary.get('small_targets')} "
            f"max_pos={summary.get('max_position_error_m')}m "
            f"max_rot={summary.get('max_rotation_error_deg')}deg "
            f"current_pos={current.get('position_error_m')}m "
            f"current_rot={current.get('rotation_error_deg')}deg"
        )


def print_robot(data: dict[str, Any]) -> None:
    validation = data.get("validation") or {}
    for mapping, sides in validation.items():
        print(f"mapping={mapping}")
        for side, rep in (sides or {}).items():
            fk = rep.get("urdf_fk_vs_sdk_error") or {}
            ik = ((rep.get("ik_current_pose_self_consistency") or {}).get("ik") or {})
            print(
                f"  {side}: "
                f"fk_pos={fk.get('position_error_m')}m "
                f"fk_rot={fk.get('rotation_error_deg')}deg "
                f"ik_success={ik.get('success')} "
                f"ik_pos={ik.get('position_error_m')}m "
                f"ik_rot={ik.get('rotation_error_deg')}deg "
                f"indices={rep.get('arm_state_indices')}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json")
    args = parser.parse_args()
    data = load_json(args.report_json)
    if "validation" in data:
        print_robot(data)
    elif "sides" in data:
        print_offline(data)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

