# 2026-06-26 G1 Artifact Management

## Standard Layout

All new validation artifacts go under:

```text
artifacts/g1_humanego/<session>/<role>/<run>/
```

Roles:

```text
server       server-side /infer request, response, RGB, visualizations
client       robot-side dry-run requests, responses, local zip
interactive  robot-side one-step execution logs
diagnostics  lower-level G1 diagnostics and transform checks
```

Default session is the current UTC date:

```text
artifacts/g1_humanego/20260626/
```

For a focused experiment, pin the session on both server and robot:

```bash
export G1_ARTIFACT_SESSION=20260626_pose_gate
```

## Naming

Server runs use the request id:

```text
server/20260626_073114_client_dry_run_000/
```

Client dry-runs use:

```text
client/client_dry_run_YYYYMMDD_HHMMSS_<tag>/
```

Interactive runs use:

```text
interactive/interactive_YYYYMMDD_HHMMSS_<tag>/
```

## Useful Commands

List artifacts:

```bash
python scripts/list_g1_artifacts.py
```

Numeric report for latest server response:

```bash
python scripts/report_g1_humanego_response.py --latest
```

Clean visualizations for latest server response:

```bash
python scripts/visualize_g1_humanego_response.py --latest --split-layers
```

This writes:

```text
response_projection_clean.jpg
response_projection_objects.jpg
response_projection_tcp.jpg
response_projection_axes.jpg
```

Stability summary:

```bash
python scripts/summarize_g1_humanego_runs.py --last 6 --recent-rgbd 3
```

## Cleanup Rule

`artifacts/` is ignored by git. Keep only runs that support a current decision.
After a validation is summarized in docs, old runs can be removed safely:

```bash
rm -rf artifacts/g1_humanego/<session>/<role>/<run>
```

Do not commit generated artifacts or zip uploads.
