#!/usr/bin/env bash
set -e

DEVICE="${1:-/dev/video0}"

echo "Camera devices:"
ls /dev/video* || true

echo
echo "Formats for ${DEVICE}:"
v4l2-ctl --device="${DEVICE}" --list-formats-ext || true

echo
echo "Capturing test frame to camera_test.jpg"
ffmpeg -y -f video4linux2 -i "${DEVICE}" -frames:v 1 camera_test.jpg

echo "Done: camera_test.jpg"
