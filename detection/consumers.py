import json
import base64
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


class DetectionConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self._face_mesh = None
        self._detector = None
        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_error = None
        await self.accept()
        logger.info("Client connected")

    async def disconnect(self, close_code):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass
        logger.info("Client disconnected")

    async def receive(self, text_data):
        try:
            msg = json.loads(text_data)
        except Exception:
            return

        if msg.get("type") != "frame":
            return

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor, self._process_frame, msg.get("data", "")
        )
        if result is not None:
            await self.send(text_data=json.dumps(result))

    # ── thread pool ────────────────────────────────────────────────

    def _ensure_initialized(self):
        with self._init_lock:
            if self._initialized:
                return
            if self._init_error:
                raise RuntimeError(self._init_error)
            try:
                logger.info("Initializing FaceMesh and Detector...")
                from core.face_mesh import FaceMesh
                from core.detector import DrowsinessDetector
                from detection.models import SystemConfig
                config = SystemConfig.get_active()
                self._face_mesh = FaceMesh(use_gpu=False)
                self._detector = DrowsinessDetector(
                    ear_threshold=config.ear_threshold,
                    mar_threshold=config.mar_threshold,
                    drowsy_frame_threshold=config.drowsy_frame_threshold,
                    yawn_frame_threshold=config.yawn_frame_threshold,
                    max_blink_frames=config.max_blink_frames,
                    perclos_window_sec=config.perclos_window_sec,
                    perclos_threshold=config.perclos_threshold,
                )
                self._initialized = True
                logger.info("Detector ready")
            except Exception as e:
                self._init_error = str(e)
                logger.error(f"Detector init failed: {e}", exc_info=True)
                raise

    def _process_frame(self, b64_data: str):
        try:
            self._ensure_initialized()
        except Exception as e:
            return {"state": "error", "message": f"Init xatosi: {e}",
                    "confidence": 0.0, "ear_avg": 0.0, "mar": 0.0,
                    "total_blinks": 0, "perclos": 0.0, "head_roll": 0.0, "timestamp": 0}

        try:
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_data)
            arr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return None

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            mesh = self._face_mesh.process(frame_rgb)
            if not mesh.has_face:
                return {
                    "state": "no_face",
                    "confidence": 0.0,
                    "ear_avg": 0.0,
                    "mar": 0.0,
                    "total_blinks": self._detector.stats.total_blinks,
                    "perclos": 0.0,
                    "head_roll": 0.0,
                    "timestamp": 0,
                }

            result = self._detector.process_frame(mesh.landmarks, w, h)
            return {
                "state": result.state.value,
                "ear_left": result.ear_left,
                "ear_right": result.ear_right,
                "ear_avg": result.ear_avg,
                "mar": result.mar,
                "total_blinks": result.total_blinks,
                "closed_frames": result.closed_frame_count,
                "perclos": result.perclos,
                "head_roll": result.head_roll,
                "confidence": result.confidence,
                "timestamp": result.timestamp,
            }
        except Exception as e:
            logger.error(f"Frame error: {e}", exc_info=True)
            return {"state": "error", "message": str(e),
                    "confidence": 0.0, "ear_avg": 0.0, "mar": 0.0,
                    "total_blinks": 0, "perclos": 0.0, "head_roll": 0.0, "timestamp": 0}
