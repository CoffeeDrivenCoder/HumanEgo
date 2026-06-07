#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download HumanEgo Aria data from the public HuggingFace dataset Leo-TX/HumanEgo.

No token / login required — the dataset is public.

Modes
-----
  test  : ONE sample recording, INPUT ONLY (~0.6 GB). Enough to run the
          preprocessing pipeline yourself and reproduce every output.
  full  : ONE sample recording, input + the precomputed preprocess output,
          including `all_data.tar` (~2 GB). The tar is auto-extracted so you can
          inspect results without running the GPU pipeline.
  all   : the ENTIRE dataset (large). Adds `--with-tar` to also pull every
          recording's `all_data.tar` (very large); otherwise the per-frame tars
          are skipped and you can regenerate them with preprocessing.

Examples
--------
    pip install huggingface_hub
    python scripts/download_data.py --mode test     # then run preprocessing
    python scripts/download_data.py --mode full     # download precomputed results
    python scripts/download_data.py --mode all --with-tar
"""
import argparse
import glob
import os
import subprocess

from huggingface_hub import snapshot_download

REPO_ID = "Leo-TX/HumanEgo"
SAMPLE = "serve_bread/aria/mps_serve_bread_000_vrs"  # the released sample recording


def extract_tars(local_dir, keep=False):
    """Unpack every preprocess/all_data.tar back into its preprocess/ folder."""
    tars = glob.glob(os.path.join(local_dir, "**", "preprocess", "all_data.tar"),
                     recursive=True)
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
    ap.add_argument("--mode", choices=["test", "full", "all"], default="test")
    ap.add_argument("--out", default="./data", help="download destination (default: ./data)")
    ap.add_argument("--with-tar", action="store_true",
                    help="[all mode] also download the large per-frame all_data.tar files")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep all_data.tar after extracting (default: delete to save space)")
    args = ap.parse_args()

    if args.mode == "test":
        allow, ignore = [f"{SAMPLE}/*"], [f"{SAMPLE}/preprocess/*"]
    elif args.mode == "full":
        allow, ignore = [f"{SAMPLE}/*"], None
    else:  # all
        allow, ignore = None, (None if args.with_tar else ["*all_data.tar"])

    print(f"Downloading [{args.mode}] from {REPO_ID} -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=allow,
        ignore_patterns=ignore,
    )

    # extract per-frame archives whenever we actually fetched them
    if args.mode == "full" or (args.mode == "all" and args.with_tar):
        extract_tars(args.out, keep=args.keep_tar)

    print("✅ done")
    if args.mode == "test":
        rec = os.path.join(args.out, SAMPLE)
        print("\nNext — run preprocessing on the sample:")
        print(f"  python -m preprocess.Preprocess --mps_path {rec} --task serve_bread")


if __name__ == "__main__":
    main()
