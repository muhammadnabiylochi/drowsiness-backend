"""
Detection Engine Singleton
===========================
Wraps core/ modules in a thread-safe singleton.
Runs the 30fps detection loop in a daemon thread.
Persists events to DB and broadcasts via Channels.
"""

import threading
import time
import logging
from typing import Optional

from core.detector import DrowsinessDetector, DriverState, DetectionResult
from core.camera import CameraStream
from core.alert import AlertManager
from core.face_mesh import FaceMesh

logger = logging.getLogger(__name__)


class DetectionEngine:
    _instance: Optional["DetectionEngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.camera: Optional[CameraStream] = None
        self.detector: Optional[DrowsinessDetector] = None
        self.alert_manager: Optional[AlertManager] = None
        self.face_mesh: Optional[FaceMesh] = None
        self.latest_result: Optional[DetectionResult] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._result_lock = threading.Lock()
        self._session_id = None
        self._prev_state = None
        self.server_start_time: float = time.time()

    def initialize(self, config):
        """Initialize all core components from SystemConfig model."""
        self.detector = DrowsinessDetector(
            ear_threshold=config.ear_threshold,
            mar_threshold=config.mar_threshold,
            drowsy_frame_threshold=config.drowsy_frame_threshold,
            yawn_frame_threshold=config.yawn_frame_threshold,
            max_blink_frames=config.max_blink_frames,
            perclos_window_sec=config.perclos_window_sec,
            perclos_threshold=config.perclos_threshold,
        )
        self.alert_manager = AlertManager(
            enabled=config.alert_enabled,
            sound_file=config.alert_sound_file or None,
            cooldown_sec=config.alert_cooldown_sec,
            webhook_url=config.alert_webhook_url or None,
        )
        self.face_mesh = FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            use_gpu=config.gpu_enabled,
        )
        self.camera = CameraStream(
            source=config.camera_source,
            width=config.camera_width,
            height=config.camera_height,
            fps=config.camera_fps,
            flip_horizontal=config.camera_flip_horizontal,
        )
        logger.info(f"Engine initialized (backend={self.face_mesh.backend})")

    def start(self):
        if self._running:
            return
        if not self.camera or not self.camera.start():
            logger.error("Camera failed to start")
            return

        from .models import DetectionSession
        session = DetectionSession.objects.create(
            camera_source=str(getattr(self.camera, '_source', '0')),
            is_active=True,
        )
        self._session_id = session.id

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Detection engine started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.camera:
            self.camera.stop()
        if self.face_mesh:
            self.face_mesh.close()
        self._close_session()
        logger.info("Detection engine stopped")

    def _close_session(self):
        if not self._session_id:
            return
        try:
            from django.utils import timezone
            from .models import DetectionSession
            session = DetectionSession.objects.get(id=self._session_id)
            session.is_active = False
            session.ended_at = timezone.now()
            if self.detector:
                s = self.detector.stats
                session.total_frames = s.total_frames
                session.total_blinks = s.total_blinks
                session.drowsy_events = s.drowsy_events
                session.yawn_events = s.yawn_events
                session.avg_ear = round(s.avg_ear, 4)
                session.avg_mar = round(s.avg_mar, 4)
                session.avg_perclos = round(s.perclos, 4)
            session.save()
        except Exception as e:
            logger.error(f"Failed to close session: {e}")

    def _loop(self):
        """Main 30fps detection loop."""
        while self._running and self.camera and self.camera.is_running:
            frame_rgb = self.camera.get_frame_rgb()
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            mesh_result = self.face_mesh.process(frame_rgb)
            if mesh_result.has_face:
                h, w = frame_rgb.shape[:2]
                result = self.detector.process_frame(mesh_result.landmarks, w, h)

                with self._result_lock:
                    self.latest_result = result

                # Alerts
                if result.state in (
                    DriverState.DROWSY, DriverState.DROWSY_YAWNING,
                    DriverState.FALLING_RIGHT, DriverState.FALLING_LEFT,
                    DriverState.FALLING_BACK,
                ):
                    self.alert_manager.trigger(
                        state=result.state.value,
                        ear_avg=result.ear_avg,
                        mar=result.mar,
                        perclos=result.perclos,
                        blinks=result.total_blinks,
                        head_roll=result.head_roll,
                        confidence=result.confidence,
                    )
                    self._save_alert(result)

                # Persist state changes
                if result.state.value != self._prev_state:
                    self._save_event(result)
                    self._prev_state = result.state.value

                # Broadcast to WebSocket
                self._broadcast(result)

            time.sleep(1 / 30)

    def _save_event(self, result: DetectionResult):
        try:
            from .models import DetectionEvent
            DetectionEvent.objects.create(
                session_id=self._session_id,
                state=result.state.value,
                previous_state=self._prev_state or "",
                ear_left=result.ear_left,
                ear_right=result.ear_right,
                ear_avg=result.ear_avg,
                mar=result.mar,
                perclos=result.perclos,
                head_roll=result.head_roll,
                head_pitch=result.head_pitch,
                confidence=result.confidence,
                total_blinks=result.total_blinks,
            )
        except Exception as e:
            logger.error(f"Event save failed: {e}")

    def _save_alert(self, result: DetectionResult):
        try:
            from .models import AlertLog
            AlertLog.objects.create(
                session_id=self._session_id,
                state=result.state.value,
                ear_avg=result.ear_avg,
                mar=result.mar,
                perclos=result.perclos,
                total_blinks=result.total_blinks,
                head_roll=result.head_roll,
                confidence=result.confidence,
                alert_number=self.alert_manager.alert_count,
            )
        except Exception as e:
            logger.error(f"Alert save failed: {e}")

    def _broadcast(self, result: DetectionResult):
        try:
            from channels.layers import get_channel_layer
            from asgiref.sync import async_to_sync
            layer = get_channel_layer()
            if layer:
                async_to_sync(layer.group_send)(
                    "detection_stream",
                    {
                        "type": "detection.update",
                        "data": {
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
                        },
                    },
                )
        except Exception:
            pass

    def get_result(self) -> Optional[DetectionResult]:
        with self._result_lock:
            return self.latest_result

    def update_thresholds(self, **kwargs):
        if not self.detector:
            return
        for key, value in kwargs.items():
            if hasattr(self.detector, key):
                setattr(self.detector, key, value)


def get_engine() -> DetectionEngine:
    return DetectionEngine()
