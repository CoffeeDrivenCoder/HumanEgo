# 2026-06-26 G1 HumanEgo RGB-D Pose Gate

## Current State

The server-client HumanEgo path is already wired:

```text
G1 RGB-D + current TCP state
-> server HumanEgo policy + RGB-D object pose
-> T_link7_target_in_base / right_pose_flat_limited
-> interactive one-step EE control
```

The important gate before further robot motion is not model loading anymore. It
is whether the server is using stable real RGB-D `obj1` / `obj2` poses instead
of fixed debug poses.

## Existing Run Evidence

Using:

```bash
python scripts/summarize_g1_humanego_runs.py --last 12 --recent-rgbd 8
```

Current saved server artifacts show:

```text
responses: 140
object_source_used:
  rgbd: 71
  fixed_fallback_after_rgbd_error: 11
```

Latest three useful RGB-D responses:

```text
20260623_100614_interactive_step_000: rgbd, raw_step=0.0252m, clipped=false
20260623_100622_interactive_step_001: rgbd, raw_step=0.0073m, clipped=false
20260623_100631_interactive_step_002: rgbd, raw_step=0.0073m, clipped=false
```

Their projected object centers are visually plausible:

```bash
python scripts/visualize_g1_humanego_response.py --latest
```

This writes:

```text
g1_humanego_server_runs/<run>/response_projection_overlay.jpg
```

For the latest saved run, `obj2` projects onto the plate area and `obj1` projects
near the right-side bread/gripper interaction area. The target TCP arrow is small,
which matches the high `done_prob` in that scene.

## Important Config Guard

`cfg/inference/g1_serve_bread_right.yaml` now has:

```yaml
perception:
  allow_fixed_object_fallback: false
```

So when the server is restarted with `G1_HUMANEGO_OBJECT_SOURCE=rgbd`, RGB-D
object pose failures should fail visibly instead of silently falling back to
fixed debug poses.

## Next Server Run

On the server:

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
git pull origin main

G1_HUMANEGO_RESTART=1 \
G1_HUMANEGO_OBJECT_SOURCE=rgbd \
bash scripts/start_g1_humanego_inference_server.sh
```

Expected health:

```text
Object source: rgbd
```

## Next Robot Dry-Run

On the robot:

```bash
cd ~/桌面/HumanEgo
git pull origin main

G1_HUMANEGO_SEND_DEPTH=true \
G1_HUMANEGO_STEPS=3 \
bash scripts/run_g1_humanego_client_dry_run_to_public_server.sh
```

Then on the server inspect:

```bash
python scripts/summarize_g1_humanego_runs.py --last 12 --recent-rgbd 6
python scripts/visualize_g1_humanego_response.py --latest
```

Proceed only if:

```text
object_source_used == rgbd
object_error is absent
obj1/obj2 projection lands on the real bread/plate
raw_delta_norm_m is small or server-clipped to <= 0.03m
```

## Next Interactive Step

Only after the RGB-D gate passes:

```bash
G1_HUMANEGO_CONFIRM=RUN_CONTROL \
G1_HUMANEGO_SEND_DEPTH=true \
G1_HUMANEGO_MAX_STEPS=3 \
G1_HUMANEGO_TARGET_SOURCE=position_keep_orientation \
G1_HUMANEGO_TARGET_ADAPTER=position_only \
G1_HUMANEGO_EXECUTE_S=1.0 \
G1_HUMANEGO_SEND_HZ=10 \
bash scripts/run_g1_humanego_interactive_step_to_public_server.sh
```

Start with `position_only` and no gripper. If observed motion direction is
correct, move to:

```text
TARGET_ADAPTER=position_orientation_limited
```

Then, only after repeated stable single steps, consider full orientation and
gripper execution.
