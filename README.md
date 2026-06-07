<p align="center">
  <a href="https://humanego-ai.github.io">
    <img src="assets/title/hero.png" alt="HumanEgo — Zero-Shot Robot Learning from Minutes of Human Egocentric Videos" width="100%" />
  </a>
</p>

<p align="center">
  <a href="https://tx-leo.github.io">Zhi (Leo) Wang</a> &nbsp;·&nbsp;
  <a href="https://bottle101.github.io/">Botao He</a> &nbsp;·&nbsp;
  <a href="https://colinyu1.github.io/">Kelin Yu</a> &nbsp;·&nbsp;
  <a href="https://sjlee.cc/">Seungjae Lee</a> &nbsp;·&nbsp;
  <a href="https://ruohangao.github.io/">Ruohan Gao</a> &nbsp;·&nbsp;
  <a href="https://furong-huang.com/">Furong Huang</a> &nbsp;·&nbsp;
  <a href="https://robotics.umd.edu/clark/faculty/350/Yiannis-Aloimonos">Yiannis Aloimonos</a>
</p>

<p align="center">
  <a href="https://humanego-ai.github.io"><img src="assets/title/btn_website.png" alt="Website" height="60" /></a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2605.24934"><img src="assets/title/btn_paper.png" alt="Paper" height="60" /></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2605.24934"><img src="assets/title/btn_arxiv.png" alt="arXiv" height="60" /></a>
  &nbsp;
  <a href="https://youtu.be/pdL46diijuY"><img src="assets/title/btn_video.png" alt="Video" height="60" /></a>
  &nbsp;
  <a href="#bibtex"><img src="assets/title/btn_bibtex.png" alt="BibTeX" height="60" /></a>
</p>

---

## Overview

<p align="center">
  <img src="assets/teaser.gif" alt="HumanEgo teaser" width="100%" />
</p>

There are three ways to use this repo, in increasing order of effort:

1. **[Quick Start](#quick-start)** — run the whole pipeline end-to-end on two
   sample recordings, as a smoke test.
2. **[Train on the HumanEgo Dataset](#train-on-the-humanego-dataset)** — download
   our full released data (with precomputed labels) and train, no hardware needed.
3. **[Train Your Own Policy](#train-your-own-policy)** — collect your own
   egocentric demonstrations with Project Aria glasses and train on them.

---

## Installation

```bash
git clone https://github.com/TX-Leo/HumanEgo.git
cd HumanEgo
conda create -n humanego python=3.11 -y
conda activate humanego
bash setup.sh
```

This installs PyTorch (with CUDA), the vision foundation models we use
(SAM 2, Grounding DINO, CoTracker, Orient-Anything V2), and the hand-tracking
methods (MediaPipe, WiLoR, HaMeR).

---

## Quick Start

The fastest way to run the whole pipeline end-to-end — download, preprocess, and
train on just a couple of recordings. The `HumanEgo` training job holds out the
first recording (`mps_serve_bread_000_vrs`) for evaluation and trains on the
rest, so download two.

**1. Download two recordings** — inputs only, ~1.2 GB

```bash
pip install huggingface_hub
python scripts/download_data.py --task serve_bread --num 2 --input-only
```

Fetches `mps_serve_bread_000_vrs` and `mps_serve_bread_001_vrs` into
`./data/serve_bread/aria/`, skipping the precomputed `preprocess/` output so you
run the pipeline yourself. See
[Train on the HumanEgo Dataset](#train-on-the-humanego-dataset) for the full
dataset and all download options.

**2. Preprocess both**

```bash
python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_000_vrs --task serve_bread
python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_001_vrs --task serve_bread
```

Regenerates each recording's `preprocess/` folder. See
[Step 2: Preprocessing](#step-2-preprocessing) for details.

**3. Train**

```bash
python -m training.FlowMatchingTrainer --task serve_bread --use_cfg --job HumanEgo
```

Trains on `mps_serve_bread_001_vrs` and evaluates on the held-out
`mps_serve_bread_000_vrs` (config: `cfg/training/serve_bread/HumanEgo.yaml`).
See [Step 3: Training](#step-3-training) for details.

---

## Train on the HumanEgo Dataset

Skip data collection entirely: download our full released dataset — raw Aria
recordings **and** the precomputed MPS + preprocess output — and train directly.
Everything is hosted on the public HuggingFace dataset
[`Leo-TX/HumanEgo`](https://huggingface.co/datasets/Leo-TX/HumanEgo), no login or
token required. We release two tasks: **`serve_bread`** and **`water_flowers`**.

### Download the full dataset

```bash
pip install huggingface_hub

# everything, both tasks, with precomputed preprocess output (large)
python scripts/download_data.py --task all --num all

# or one task at a time
python scripts/download_data.py --task serve_bread   --num all
python scripts/download_data.py --task water_flowers --num all
```

Each recording lands at `./data/<task>/aria/mps_<task>_<id>_vrs/` with its
`preprocess/` folder already populated (the `all_data.tar` is auto-extracted).
Use `--num N` for the first N recordings, or `--input-only` to skip the
precomputed output and run preprocessing yourself. See
[`preprocess/README.md`](preprocess/README.md) for the full output-file
reference and a plain-`huggingface_hub` recipe.

### Train

```bash
# serve_bread
python -m training.FlowMatchingTrainer --task serve_bread   --use_cfg --job HumanEgo

# water_flowers
python -m training.FlowMatchingTrainer --task water_flowers --use_cfg --job HumanEgo
```

Each job holds out recording `000` of the task for evaluation and trains on the
rest, reading `cfg/training/<task>/HumanEgo.yaml`. See
[Step 3: Training](#step-3-training) for the `--task` / `--job` convention.

---

## Train Your Own Policy

Collect your own human-egocentric demonstrations and train a policy on them,
end-to-end — record with Project Aria glasses, process the data through MPS,
preprocess it, train, and deploy.

### Step 1: Data Collection

<p align="center">
  <img src="assets/data_collection.gif" alt="HumanEgo data collection — anyone, anytime, anywhere, with only 30 minutes of data" width="100%" />
</p>

To apply for the Meta Project Aria glasses, see
[projectaria.com/glasses](https://www.projectaria.com/glasses/).

See [`datacollection/README.md`](datacollection/README.md)
for the end-to-end guide on recording your own Project Aria data and running
MPS (SLAM + hand tracking) on it. The resulting data should look like this:

```
- data
    - mps_TEST_vrs/
        - else
            - sample.vrs.json
            - vrs_health_check.json
            - vrs_health_check_slam.json
        - hand_tracking
            - hand_tracking_results.csv
            - summary.json
        - slam
            - closed_loop_trajectory.csv
            - online_calibration.jsonl
            - open_loop_trajectory.csv
            - semidense_observations.csv.gz
            - semidense_points.csv.gz
            - summary.json
        - sample.vrs
```

### Step 2: Preprocessing

<p align="center">
  <img src="assets/data_collection.webp" alt="HumanEgo preprocessing visualization" width="100%" />
</p>

Turn raw MPS output into training-ready data — point `--mps_path` at the MPS
folder from Step 1:

```bash
python -m preprocess.Preprocess --mps_path ./data/<your_mps_folder> --task <your_task>
```

This regenerates everything under `…/preprocess/`. See
**[`preprocess/README.md`](preprocess/README.md)** for the full data layout,
the output-file reference, and download options.

### Step 3: Training

<p align="center">
  <img src="assets/architecture.webp" alt="HumanEgo training architecture" width="100%" />
</p>

```bash
python -m training.FlowMatchingTrainer --task "YOUR_TASK" --use_cfg --job "YOUR_JOB"
```

`--task` selects a folder under `cfg/training/` (e.g. `serve_bread`) and
`--job` selects a YAML inside it (e.g. `HumanEgo`, resolving to
`cfg/training/serve_bread/HumanEgo.yaml`).

### Step 4: Inference

> **TODO** — documentation coming soon.

---

## TODO

We are actively releasing the following — check back soon.

- [ ] Release a 3-minute quick-start tutorial
- [ ] Release a pretrained model (for inference demo)
- [ ] Release documentation for **Training**
- [ ] Release documentation for **Inference**

---

## Acknowledgements

This project builds on excellent open-source work, including
[Project Aria](https://www.projectaria.com/) (Gen 1 glasses &amp;
[MPS](https://facebookresearch.github.io/projectaria_tools/docs/intro)),
[Trossen Arm](https://www.trossenrobotics.com/),
[CoTracker3](https://github.com/facebookresearch/co-tracker),
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM 2](https://github.com/facebookresearch/sam2),
[HaMeR](https://github.com/geopavlakos/hamer),
[WiLoR](https://github.com/rolpotamias/WiLoR),
[MediaPipe](https://github.com/google-ai-edge/mediapipe),
[LaMa](https://github.com/advimman/lama),
and [Orient-Anything](https://github.com/SpatialVision/Orient-Anything).

---

<h2 id="bibtex">BibTeX</h2>

```bibtex
@misc{humanego2026,
  title         = {HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos},
  author        = {Wang, Zhi and He, Botao and Yu, Kelin and Lee, Seungjae and Gao, Ruohan and Huang, Furong and Aloimonos, Yiannis},
  year          = {2026},
  eprint        = {2605.24934},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
