#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download HumanEgo Aria data from the public HuggingFace dataset Leo-TX/HumanEgo.

No token / login required — the dataset is public.

Pick a task and how many recordings:
  --task   serve_bread | water_flowers | all     (default: serve_bread)
  --num    N  or  all                             (default: 1 — the first N recordings)
  --input-only   download just the inputs (so you can run preprocessing yourself).
                 Otherwise the precomputed `preprocess/` output is included and each
                 recording's `all_data.tar` is auto-extracted.
  --out    download destination                   (default: ./data)

Examples
--------
    pip install huggingface_hub

    python scripts/download_data.py                                 # 1 serve_bread recording, full output
    python scripts/download_data.py --task serve_bread --num 20     # first 20 serve_bread, with output
    python scripts/download_data.py --task all       --num all      # the entire dataset (large)
    python scripts/download_data.py --task water_flowers --num 5 --input-only   # 5 inputs, to run yourself
"""
import argparse
import glob
import os
import subprocess

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Leo-TX/HumanEgo"
TASKS = ["serve_bread", "water_flowers"]


def list_recordings(task):
    """Sorted recording dirs for a task, e.g. 'serve_bread/aria/mps_serve_bread_000_vrs'."""
    try:
        tree = HfApi().list_repo_tree(REPO_ID, path_in_repo=f"{task}/aria",
                                      repo_type="dataset", recursive=False)
        # list_repo_tree is lazy — iterate INSIDE the try so a missing path (404) is caught
        return sorted(e.path for e in tree if e.path.endswith("_vrs"))
    except Exception:
        return []


def extract_tars(local_dir, keep=False):
    """Unpack every preprocess/all_data.tar back into its preprocess/ folder."""
    tars = glob.glob(os.path.join(local_dir, "**", "preprocess", "all_data.tar"), recursive=True)
    for tar in tars:
        out = os.path.dirname(tar)
        print(f"  extracting {os.path.relpath(tar, local_dir)}")
        subprocess.run(["tar", "-xf", tar, "-C", out], check=True)
        if not keep:
            os.remove(tar)
    if tars:
        print(f"  extracted {len(tars)} all_data.tar archive(s)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=TASKS + ["all"], default="serve_bread")
    ap.add_argument("--num", default="1",
                    help="how many recordings (first N), or 'all' (default: 1)")
    ap.add_argument("--input-only", action="store_true",
                    help="download inputs only (run preprocessing yourself); skip precomputed output")
    ap.add_argument("--out", default="./data", help="download destination (default: ./data)")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep all_data.tar after extracting (default: delete to save space)")
    args = ap.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    num = None if str(args.num).lower() == "all" else int(args.num)

    recs = []
    for t in tasks:
        found = list_recordings(t)
        if not found:
            print(f"  [warn] no recordings found for task '{t}' (not uploaded yet?)")
        recs += found if num is None else found[:num]
    if not recs:
        raise SystemExit("Nothing to download.")

    allow = [f"{r}/*" for r in recs]                       # '*' spans '/' on the Hub -> recursive
    ignore = ["*/preprocess/*"] if args.input_only else None

    kind = "input only" if args.input_only else "input + precomputed output"
    print(f"Downloading {len(recs)} recording(s) [{args.task}] ({kind}) -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=args.out,
        allow_patterns=allow, ignore_patterns=ignore,
    )

    if not args.input_only:
        extract_tars(args.out, keep=args.keep_tar)

    print("✅ done")
    if args.input_only:
        task0 = recs[0].split("/")[0]
        print(f"\nNext — run preprocessing, e.g.:\n"
              f"  python -m preprocess.Preprocess --mps_path {os.path.join(args.out, recs[0])} --task {task0}")


if __name__ == "__main__":
    main()
