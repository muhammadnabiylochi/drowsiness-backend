"""
Camera Module
==============
Abstraction layer for video input sources.
Supports: local webcam, RTSP/HTTP IP cameras (Axis P1367 etc.), video files.

Paper setup (Section 4.1, Figure 4):
    - Camera distance from eyes: 50cm (0.3m - 0.6m range)
    - Angle view: frontal
"""

import time
import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """
    Thread-safe camera stream reader.
    
    Reads frames in a background thread to avoid blocking
    the main processing loop (important for real-time performance
    on Jetson Orin Nano).
    """

    def __init__(
        self,
        source: str = "0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = True,
    ):
        self.source = int(source) if source.isdigit() else source
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip_horizontal

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._ret: bool = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self) -> bool:
        """Initialize camera and start capture thread."""
        logger.info(f"Opening camera: {self.source}")

        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open camera: {self.source}")
            return False

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # For IP cameras: set buffer size to reduce latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def _capture_loop(self):
        """Background frame capture."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                if self.flip:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = frame
                    self._ret = True
                    self._frame_count += 1
            else:
                # Reconnect for IP cameras
                if isinstance(self.source, str) and ('rtsp' in self.source or 'http' in self.source):
                    logger.warning("Camera disconnected, reconnecting...")
                    time.sleep(1)
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.source)
                else:
                    time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame (non-blocking)."""
        with self._lock:
            if self._frame is not None:
                return self._ret, self._frame.copy()
            return False, None

    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """Get latest frame in RGB format (for MediaPipe)."""
        ret, frame = self.read()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
