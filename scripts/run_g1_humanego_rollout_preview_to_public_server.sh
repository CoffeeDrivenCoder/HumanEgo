#!/usr/bin/env bash
set -euo pipefail

# Robot-side rollout preview. Sends one RGB-D frame and current robot state to
# the HumanEgo server, asks it to fix the first-frame objects, and autoregressively
# generate a target trajectory. This script sends no robot control commands.

cd "$(dirname "$0")/.."

export G1_HUMANEGO_TAG="${G1_HUMANEGO_TAG:-rollout_preview}"
export G1_HUMANEGO_STEPS="${G1_HUMANEGO_STEPS:-1}"
export G1_HUMANEGO_PREVIEW_STEPS="${G1_HUMANEGO_PREVIEW_STEPS:-1}"
export G1_HUMANEGO_ROLLOUT_STEPS="${G1_HUMANEGO_ROLLOUT_STEPS:-20}"
export G1_HUMANEGO_ROLLOUT_TARGET_SOURCE="${G1_HUMANEGO_ROLLOUT_TARGET_SOURCE:-raw}"
export G1_HUMANEGO_ROLLOUT_UPDATE_GRIPPER="${G1_HUMANEGO_ROLLOUT_UPDATE_GRIPPER:-true}"
export G1_HUMANEGO_SEND_DEPTH="${G1_HUMANEGO_SEND_DEPTH:-true}"
export G1_HUMANEGO_SEND_WIDTH="${G1_HUMANEGO_SEND_WIDTH:-640}"
export G1_HUMANEGO_SEND_HEIGHT="${G1_HUMANEGO_SEND_HEIGHT:-400}"
export G1_HUMANEGO_JPEG_QUALITY="${G1_HUMANEGO_JPEG_QUALITY:-75}"

bash scripts/run_g1_humanego_client_dry_run_to_public_server.sh "$@"
