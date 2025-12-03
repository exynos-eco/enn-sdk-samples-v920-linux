#!/bin/bash

TEST_INDEX=${1:-1}

BASE_PATH="/data/vendor/esrgan_x4plus"
BIN_NAME="enn_sample_esrgan_x4plus"
INPUT_FILE="media/image.jpg"

# WESTON_CMD="weston --backend=drm-backend.so --idle-time=0 --tty=1"
WESTON_CMD="weston --backend=drm-backend.so --idle-time=0 --tty=1 > /tmp/weston.log 2>&1 &"
WAYLAND_SETUP_CMD="export WAYLAND_DISPLAY=wayland-1"

RUN_CMD="sh -c '${WESTON_CMD} & sleep 2; ${WAYLAND_SETUP_CMD}; ${BASE_PATH}/${BIN_NAME} -c ${TEST_INDEX} -i ${BASE_PATH}/${INPUT_FILE}'"

echo "RUN_CMD : $RUN_CMD"
adb shell "$RUN_CMD"