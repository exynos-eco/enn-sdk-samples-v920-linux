#!/bin/bash

echo "[INFO] Checking processes using /dev/dri/card0..."
PIDS=$(fuser /dev/dri/card0 2>/dev/null)

if [ -n "$PIDS" ]; then
    echo "[INFO] Found PID(s): $PIDS"

    TARGET_PIDS=$(echo "$PIDS" | awk '{for (i=1; i<=NF; i+=2) printf "%s ", $i}')
    echo "[INFO] Killing odd-indexed PIDs: $TARGET_PIDS"

    for PID in $TARGET_PIDS; do
        kill -9 "$PID" 2>/dev/null && echo "[INFO] Killed PID $PID" || echo "[WARN]"
    done

    sleep 1
else
    echo "[INFO] No process is currently using /dev/dri/card0"
fi

echo "[INFO] Starting weston..."
weston --backend=drm-backend.so --idle-time=0 --debug > /tmp/weston.log 2>&1 &
WESTON_PID=$!
export WAYLAND_DISPLAY=wayland-1
sleep 2

if ps -p "$WESTON_PID" > /dev/null; then
    echo "[INFO] weston is running (PID=$WESTON_PID)"
else
    echo "[ERROR] weston failed to start."
    echo "[HINT] Check log: /tmp/weston.log"
    exit 1
fi
