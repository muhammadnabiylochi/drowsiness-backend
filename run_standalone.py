#!/usr/bin/env python3
"""
Standalone Drowsiness Detector (CLI Mode)
==========================================
Runs detection directly with OpenCV window output.
No web server required — ideal for Jetson Orin Nano deployment.

Usage:
    python run_standalone.py                          # Webcam
    python run_standalone.py --source 0               # Webcam (explicit)
    python run_standalone.py --source rtsp://...      # IP camera
    python run_standalone.py --source video.mp4       # Video file
    python run_standalone.py --ear-threshold 0.20     # Custom EAR
    python run_standalone.py --headless                # No GUI (server/edge)
"""

import argparse
import time
import sys
import logging

import cv2
import numpy as np

from core.detector import DrowsinessDetector, DriverState, LEFT_EYE_IDX, RIGHT_EYE_IDX
from core.camera import CameraStream
from core.alert import AlertManager
from core.face_mesh import FaceMesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_AMBER = (0, 165, 255)
COLOR_CYAN = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)


def draw_landmarks_on_frame(frame, landmarks, w, h):
    """Draw eye and mouth landmarks on frame."""
    # Eye landmarks
    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, COLOR_CYAN, -1)

    # Mouth landmarks
    for idx in [13, 14, 78, 308]:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, COLOR_AMBER, -1)


def draw_ear_graph(frame, ear_history, threshold, x_offset, y_offset, graph_w, graph_h):
    """Draw EAR waveform on frame (like paper Figure 6/7)."""
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + graph_w, y_offset + graph_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Border
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + graph_w, y_offset + graph_h), (50, 50, 50), 1)

    if len(ear_history) < 2:
        return

    # Threshold line
    t_y = int(y_offset + graph_h - (threshold / 0.5) * graph_h)
    cv2.line(frame, (x_offset, t_y), (x_offset + graph_w, t_y), COLOR_GREEN, 1)
    cv2.putText(frame, f"T={threshold:.2f}", (x_offset + 4, t_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GREEN, 1)

    # Waveform
    max_points = min(len(ear_history), graph_w)
    start = len(ear_history) - max_points
    points = []
    for i in range(max_points):
        x = x_offset + int(i * graph_w / max_points)
        val = ear_history[start + i]
        y = int(y_offset + graph_h - (val / 0.5) * graph_h)
        y = max(y_offset, min(y_offset + graph_h, y))
        points.append((x, y))

    for i in range(1, len(points)):
        color = COLOR_RED if ear_history[start + i] < threshold else COLOR_CYAN
        cv2.line(frame, points[i - 1], points[i], color, 1)

    cv2.putText(frame, "EAR", (x_offset + 4, y_offset + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)


def main():
    parser = argparse.ArgumentParser(description="Standalone Drowsiness Detector")
    parser.add_argument("--source", default="0", help="Camera source")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ear-threshold", type=float, default=0.22)
    parser.add_argument("--mar-threshold", type=float, default=0.55)
    parser.add_argument("--drowsy-frames", type=int, default=15)
    parser.add_argument("--headless", action="store_true", help="No GUI window")
    parser.add_argument("--alert-sound", default=None, help="Path to alert .wav file")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable GPU acceleration (default)")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", help="Force CPU mode")
    args = parser.parse_args()

    # Initialize
    detector = DrowsinessDetector(
        ear_threshold=args.ear_threshold,
        mar_threshold=args.mar_threshold,
        drowsy_frame_threshold=args.drowsy_frames,
    )

    alert = AlertManager(
        enabled=True,
        sound_file=args.alert_sound,
        cooldown_sec=5.0,
    )

    # MediaPipe (GPU-accelerated if available)
    face_mesh = FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        use_gpu=args.gpu,
    )

    # Camera
    cam = CameraStream(
        source=args.source,
        width=args.width,
        height=args.height,
    )

    if not cam.start():
        logger.error("Cannot open camera")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  DROWSINESS DETECTOR — Standalone Mode")
    print(f"  Source: {args.source}")
    print(f"  EAR threshold: {args.ear_threshold}")
    print(f"  Backend: {face_mesh.backend}")
    print(f"  Press 'q' to quit")
    print("=" * 50 + "\n")

    fps_timer = time.time()
    frame_count = 0

    try:
        while True:
            frame_rgb = cam.get_frame_rgb()
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            h, w = frame_rgb.shape[:2]

            # Face Mesh (GPU-accelerated if available)
            mesh_result = face_mesh.process(frame_rgb)

            ret, display_frame = cam.read()
            if not ret or display_frame is None:
                continue

            if mesh_result.has_face:
                landmarks = mesh_result.landmarks

                # Detect
                result = detector.process_frame(landmarks, w, h)

                if not args.headless:
                    # Draw landmarks
                    draw_landmarks_on_frame(display_frame, landmarks, w, h)

                    # Status
                    state_text = result.state.value.upper()
                    if result.state == DriverState.DROWSY:
                        state_color = COLOR_RED
                    elif result.state == DriverState.YAWNING:
                        state_color = COLOR_AMBER
                    else:
                        state_color = COLOR_GREEN

                    cv2.putText(display_frame, state_text, (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)

                    # Metrics
                    metrics = [
                        f"EAR: {result.ear_avg:.3f}",
                        f"MAR: {result.mar:.3f}",
                        f"Blinks: {result.total_blinks}",
                        f"PERCLOS: {result.perclos:.2f}",
                        f"Roll: {result.head_roll:.1f}deg",
                    ]
                    for i, txt in enumerate(metrics):
                        cv2.putText(display_frame, txt, (10, 65 + i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

                    # EAR waveform graph
                    draw_ear_graph(
                        display_frame,
                        detector.get_ear_history(),
                        detector.ear_threshold,
                        x_offset=w - 220,
                        y_offset=10,
                        graph_w=200,
                        graph_h=100,
                    )

                # Alerts
                if result.state in (DriverState.DROWSY, DriverState.DROWSY_YAWNING):
                    alert.trigger(
                        state=result.state.value,
                        ear_avg=result.ear_avg,
                        mar=result.mar,
                        perclos=result.perclos,
                        blinks=result.total_blinks,
                    )

            else:
                if not args.headless:
                    cv2.putText(display_frame, "NO FACE", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()
                if not args.headless:
                    cv2.putText(display_frame, f"FPS: {fps:.0f}", (w - 100, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

            if not args.headless:
                cv2.imshow("Drowsiness Detection", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.stop()
        face_mesh.close()
        if not args.headless:
            cv2.destroyAllWindows()

        # Print summary
        s = detector.stats
        uptime = time.time() - s.session_start
        print("\n" + "=" * 50)
        print("  SESSION SUMMARY")
        print(f"  Duration:     {uptime:.0f} seconds")
        print(f"  Frames:       {s.total_frames}")
        print(f"  Avg FPS:      {s.total_frames / max(uptime, 0.001):.1f}")
        print(f"  Total blinks: {s.total_blinks}")
        print(f"  Drowsy events:{s.drowsy_events}")
        print(f"  Yawn events:  {s.yawn_events}")
        print(f"  Avg EAR:      {s.avg_ear:.4f}")
        print(f"  PERCLOS:      {s.perclos:.4f}")
        print(f"  Alerts:       {alert.alert_count}")
        print("=" * 50)


if __name__ == "__main__":
    main()
