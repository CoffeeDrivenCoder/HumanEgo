#!/usr/bin/env python3
"""Export one HumanEgo serve_bread recording to a LeRobot v3 EE-action dataset.

This is intentionally a small pilot exporter:
- one episode from one HumanEgo MPS folder
- head-view video frames from an existing rendered MP4
- EE pose actions from HumanEgo hand trajectories, no robot joint targets
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MPS_PATH = Path("/data/wangk/data/serve_bread/aria/mps_serve_bread_006_vrs")
DEFAULT_RENDER_VIDEO = Path(
    "/home/ubuntu/projects/wangk/HumanEgo/outputs/render_g1_phantom_pipeline/"
    "g1_phantom_multi_sleeve_color_006_full.mp4"
)
DEFAULT_OUT_ROOT = Path("/home/ubuntu/projects/wangk/HumanEgo/outputs/lerobot_humanego_ee_pilot")
DEFAULT_LEROBOT_SRC = Path("/home/ubuntu/projects/lerobot-for-umi/src")


def rotmat_to_o6d(rot: np.ndarray) -> np.ndarray:
    """Flatten the first two columns of a rotation matrix."""
    return np.asarray(rot, dtype=np.float32).reshape(3, 3)[:, :2].reshape(-1)


def pose_to_ee_vec(transform: list[list[float]], grasp: float) -> np.ndarray:
    pose = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return np.concatenate(
        [
            pose[:3, 3].astype(np.float32),
            rotmat_to_o6d(pose[:3, :3]),
            np.asarray([float(grasp)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def load_labeled_frames(mps_path: Path) -> list[tuple[int, Path]]:
    all_data = mps_path / "preprocess" / "all_data"
    frames = []
    for frame_dir in sorted(p for p in all_data.iterdir() if p.is_dir()):
        label_path = frame_dir / "training_data.json"
        if label_path.exists():
            frames.append((int(frame_dir.name), label_path))
    if not frames:
        raise RuntimeError(f"No labeled frames found under {all_data}")
    return frames


def read_pose(label_path: Path) -> np.ndarray:
    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    hand = data["entities"]["hands"]["right"]
    return pose_to_ee_vec(hand["T_hand_to_world"], hand.get("grasp", 0.0))


def seek_video_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read rendered video frame {frame_index}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_features(height: int, width: int) -> dict:
    return {
        "observation.images.head": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": {"axes": ["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grasp"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": {"axes": ["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grasp"]},
        },
        "source.frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
    }


def add_lerobot_to_path(lerobot_src: Path) -> None:
    src = str(lerobot_src)
    if src not in sys.path:
        sys.path.insert(0, src)


def export(args: argparse.Namespace) -> Path:
    add_lerobot_to_path(args.lerobot_src)
    from lerobot.datasets import LeRobotDataset

    frames = load_labeled_frames(args.mps_path)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    cap = cv2.VideoCapture(str(args.render_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open rendered video: {args.render_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or args.fps
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_source_idx = frames[-1][0]
    if max_source_idx >= total_video_frames:
        raise RuntimeError(
            f"Rendered video has {total_video_frames} frames, but labeled source frame {max_source_idx} is required"
        )

    ee = [read_pose(label_path) for _, label_path in frames]
    actions = ee[1:] + [ee[-1]]

    dataset_root = args.out_root / args.dataset_name
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=f"local/{args.dataset_name}",
        root=dataset_root,
        robot_type="g1_ee_humanego",
        fps=fps,
        features=build_features(height, width),
        use_videos=True,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
    )

    for (source_idx, _), state, action in zip(frames, ee, actions, strict=True):
        dataset.add_frame(
            {
                "task": args.task,
                "observation.images.head": seek_video_frame(cap, source_idx),
                "observation.state": state,
                "action": action,
                "source.frame_index": np.asarray([source_idx], dtype=np.int64),
            }
        )
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    cap.release()

    # Readback sanity check through the public LeRobot loader.
    loaded = LeRobotDataset(repo_id=f"local/{args.dataset_name}", root=dataset_root)
    sample0 = loaded[0]
    sample_last = loaded[len(loaded) - 1]
    print(f"wrote={dataset_root}")
    print(f"frames={len(loaded)} fps={fps} image={width}x{height}")
    print(f"source_frame_range={int(sample0['source.frame_index'].item())}-{int(sample_last['source.frame_index'].item())}")
    print(f"state_shape={tuple(sample0['observation.state'].shape)} action_shape={tuple(sample0['action'].shape)}")
    return dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-path", type=Path, default=DEFAULT_MPS_PATH)
    parser.add_argument("--render-video", type=Path, default=DEFAULT_RENDER_VIDEO)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dataset-name", default="serve_bread_006_ee_g1_head")
    parser.add_argument("--task", default="serve bread")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--lerobot-src", type=Path, default=DEFAULT_LEROBOT_SRC)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
