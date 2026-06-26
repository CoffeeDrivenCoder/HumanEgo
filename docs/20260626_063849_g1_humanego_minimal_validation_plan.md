# 2026-06-26 G1 HumanEgo Minimal Validation Plan

Goal: use the fewest robot/server validations that still produce decisive
information. Each validation should either unlock the next stage or identify one
specific class of problem: geometry, perception, policy output, or robot
execution.

## Artifact Layout

All new G1/HumanEgo validation artifacts should use:

```text
artifacts/g1_humanego/<session>/<role>/<run>/
```

Default session is the current UTC date, for example:

```text
artifacts/g1_humanego/20260626/server/
artifacts/g1_humanego/20260626/client/
artifacts/g1_humanego/20260626/interactive/
```

To pin a whole experiment to one session name:

```bash
export G1_ARTIFACT_SESSION=20260626_pose_gate
```

List current artifacts:

```bash
python scripts/list_g1_artifacts.py
```

Run directories are ignored by git and should not be committed.

## Principle

Do not start by limiting or rewriting the model output. First prove that the
geometry is correct, then inspect the model raw target, then execute one raw
target only if the printed numbers and visualization are plausible.

The validation order is:

```text
1. One RGB-D dry-run snapshot: prove current gripper/TCP + object geometry.
2. Short static RGB-D sequence: prove pose stability.
3. One raw model-output dry-run: prove policy target is numerically plausible.
4. One raw/full interactive robot step: prove robot motion matches target.
```

If any stage fails, stop and fix that layer before continuing.

## Validation 1: Single Geometry Snapshot

Purpose:

```text
Verify that camera intrinsics/extrinsics, robot FK/TCP, and RGB-D object pose
are mutually consistent in one real scene.
```

Run on server:

```bash
cd /home/ubuntu/projects/wangk/HumanEgo
G1_HUMANEGO_RESTART=1 \
G1_HUMANEGO_OBJECT_SOURCE=rgbd \
bash scripts/start_g1_humanego_inference_server.sh
```

Run on robot:

```bash
cd ~/桌面/HumanEgo
G1_HUMANEGO_SEND_DEPTH=true \
G1_HUMANEGO_STEPS=1 \
bash scripts/run_g1_humanego_client_dry_run_to_public_server.sh
```

Inspect on server:

```bash
python scripts/report_g1_humanego_response.py --latest
python scripts/visualize_g1_humanego_response.py --latest --split-layers
python scripts/report_g1_object_debug.py --latest
python scripts/summarize_g1_humanego_runs.py --last 3 --recent-rgbd 3
```

Pass criteria:

```text
object_source_used == rgbd
object_error is absent
current_tcp projection lands on the real right gripper/TCP area
obj1 projection lands on bread
obj2 projection lands on plate
current TCP camera Z and object camera Z are physically plausible
```

Useful failure meaning:

```text
TCP projection wrong        -> T_base_camera / FK / T_tcp_in_link7 problem.
obj projection wrong        -> RGB-D object pose / prompt / depth lifting problem.
source is fixed/fallback    -> server not truly testing perception.
```

If object projection is wrong, inspect:

```bash
python scripts/report_g1_object_debug.py --latest
```

and open:

```text
object_debug/obj1_mask_overlay.jpg
object_debug/obj1_valid_depth_overlay.jpg
object_debug/obj2_mask_overlay.jpg
object_debug/obj2_valid_depth_overlay.jpg
```

Stop condition:

```text
Do not move the robot if this fails.
```

## Validation 2: Static Pose Stability

Purpose:

```text
Verify that the same still scene produces stable object and TCP poses across a
few frames.
```

Run on robot without moving the scene:

```bash
G1_HUMANEGO_SEND_DEPTH=true \
G1_HUMANEGO_STEPS=3 \
G1_HUMANEGO_SLEEP_S=0.5 \
bash scripts/run_g1_humanego_client_dry_run_to_public_server.sh
```

Inspect on server:

```bash
python scripts/summarize_g1_humanego_runs.py --last 6 --recent-rgbd 3
python scripts/visualize_g1_humanego_response.py --latest --split-layers
```

Pass criteria:

```text
all three new responses use rgbd
no object_error
obj1/obj2 camera XYZ jitter is small relative to object size
no single-frame jump that changes which physical object is selected
```

Practical threshold:

```text
obj1 step delta max <= about 0.03m
obj2 step delta max <= about 0.03m
```

Useful failure meaning:

```text
Large jitter with correct projection -> segmentation/depth pose is unstable.
Object identity jumps                -> prompts/masks are ambiguous.
Intermittent errors                  -> depth points or model cache path need fixing.
```

Stop condition:

```text
Do not move the robot if the object identity or depth pose is unstable.
```

## Validation 3: Raw Policy Target Dry-Run

Purpose:

```text
Inspect the unmodified model target before robot execution.
```

Use the same dry-run response and inspect:

```text
current TCP pose
raw target TCP / link7 pose
translation delta: dx, dy, dz, norm
rotation delta: axis/angle
gripper raw target
target projection arrow in RGB
done probability
```

Pass criteria:

```text
translation is plausible for the scene
rotation is plausible for the scene
target is not below/inside the table or outside reachable space
target projection direction agrees with the intended approach
gripper target is not surprising for the current task phase
```

Decision rule:

```text
Do not clip or adapt the model output for this validation.
If the raw target is obviously unreasonable, do not execute; fix policy input,
object poses, frame alignment, or checkpoint assumptions first.
```

Useful failure meaning:

```text
Good geometry + bad raw target -> policy input semantics / T_align / task phase issue.
```

## Validation 4: One Raw/Full Robot Step

Purpose:

```text
Verify that the robot moves in correspondence with the raw model target.
```

Run on robot only after manually accepting Validation 1-3:

```bash
G1_HUMANEGO_CONFIRM=RUN_CONTROL \
G1_HUMANEGO_SEND_DEPTH=true \
G1_HUMANEGO_MAX_STEPS=1 \
G1_HUMANEGO_TARGET_SOURCE=raw \
G1_HUMANEGO_TARGET_ADAPTER=full \
G1_HUMANEGO_EXECUTE_S=1.0 \
G1_HUMANEGO_SEND_HZ=10 \
bash scripts/run_g1_humanego_interactive_step_to_public_server.sh
```

Before pressing Enter, read:

```text
target_delta_m
target_rotation_delta_deg
server raw_delta_norm_m
gripper target raw
right_pose
axis/object alignment printout
```

Execute only if the numbers are physically plausible.

Pass criteria:

```text
observed_delta direction matches target_delta direction
observed_delta magnitude is in the same order as target_delta
observed_rotation_delta direction/order matches target rotation
robot does not move in an unexpected coordinate direction
no unexpected gripper execution is performed
```

Useful failure meaning:

```text
Target print plausible + robot moves wrong direction -> G1 control frame or pose format issue.
Position correct + orientation wrong                 -> quaternion/frame alignment issue.
Robot barely moves                                   -> controller lifetime/send rate or SDK command issue.
```

## Validation 5: Repeat Only If Needed

If Validation 4 passes once, repeat at most 2-3 more single steps with raw/full
targets. Do not switch to continuous closed-loop until single-step target and
observed motion agree consistently.

Only after repeated single-step agreement:

```text
consider gripper execution
consider continuous low-frequency loop
consider full serve_bread rollout
```

## Minimum Data To Save Per Validation

For each meaningful run, keep:

```text
request_summary.json
response.json / server_response.json
RGB projection overlay
interactive_step_report.json for executed steps
```

Do not commit run directories to git. They are ignored by `.gitignore`.
