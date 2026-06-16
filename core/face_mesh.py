"""
GPU-Accelerated Face Mesh
==========================
Face mesh inference with GPU support via ONNX Runtime DirectML.

Initialization order:
    1. ONNX Runtime GPU (DirectML — works with NVIDIA/AMD/Intel on Windows)
    2. MediaPipe Tasks API CPU
    3. Legacy MediaPipe Solutions API (final fallback)
"""

import logging
import urllib.request
import zipfile
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = Path(__file__).parent.parent / "models"
TASK_PATH = MODEL_DIR / "face_landmarker.task"
DET_ONNX = MODEL_DIR / "face_detector.onnx"
LM_ONNX = MODEL_DIR / "face_landmarks.onnx"


@dataclass
class FaceMeshResult:
    """Unified result from face mesh processing."""
    landmarks: Optional[list]
    has_face: bool


class _Landmark:
    """Lightweight landmark with .x .y .z — compatible with MediaPipe."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


# ═══════════════════════════════════════════════════════════════════
#  BlazeFace anchor generation (matches MediaPipe short-range model)
# ═══════════════════════════════════════════════════════════════════

def _generate_anchors(input_size: int = 128) -> np.ndarray:
    """Generate SSD anchors for BlazeFace short-range detector."""
    strides = [8, 16]
    anchors_per_stride = {8: 2, 16: 6}
    anchors = []
    for stride in strides:
        grid = input_size // stride
        repeats = anchors_per_stride[stride]
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(repeats):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  ONNX Runtime GPU backend (DirectML)
# ═══════════════════════════════════════════════════════════════════

class _FaceMeshONNX:
    """
    GPU-accelerated face mesh via ONNX Runtime.

    Pipeline:
        1. BlazeFace short-range face detection  (128x128)
        2. Face landmark regression              (256x256)
    """

    DET_SIZE = 128
    LM_SIZE = 256
    DET_SCORE_THRESH = 0.5
    FACE_PRESENCE_THRESH = 0.5
    NUM_LANDMARKS = 478

    def __init__(self):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            ep = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self._provider_name = "DirectML (GPU)"
        elif "CUDAExecutionProvider" in providers:
            ep = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._provider_name = "CUDA (GPU)"
        else:
            ep = ["CPUExecutionProvider"]
            self._provider_name = "CPU"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._det = ort.InferenceSession(str(DET_ONNX), opts, providers=ep)
        self._lm = ort.InferenceSession(str(LM_ONNX), opts, providers=ep)
        self._anchors = _generate_anchors(self.DET_SIZE)

        # Cache input names
        self._det_input = self._det.get_inputs()[0].name
        self._lm_input = self._lm.get_inputs()[0].name

        # Track face region for faster re-detection
        self._prev_box = None

        logger.info(f"ONNX face mesh initialized ({self._provider_name})")

    @property
    def provider(self) -> str:
        return self._provider_name

    # ── detection ──────────────────────────────────────────────────

    def _preprocess_det(self, frame: np.ndarray) -> np.ndarray:
        """Resize to 128x128 and normalize to [-1, 1]."""
        import cv2
        img = cv2.resize(frame, (self.DET_SIZE, self.DET_SIZE))
        return (img.astype(np.float32) / 127.5 - 1.0)[np.newaxis]

    def _decode_detections(self, regressors, scores):
        """Decode BlazeFace output → best face [x1, y1, x2, y2] normalized."""
        raw = np.clip(scores[0, :, 0], -80, 80)
        scores = 1.0 / (1.0 + np.exp(-raw))  # sigmoid
        mask = scores > self.DET_SCORE_THRESH
        if not np.any(mask):
            return None

        best = np.argmax(scores * mask)
        reg = regressors[0, best]

        cx = self._anchors[best, 0] + reg[0] / self.DET_SIZE
        cy = self._anchors[best, 1] + reg[1] / self.DET_SIZE
        w = reg[2] / self.DET_SIZE
        h = reg[3] / self.DET_SIZE

        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    # ── landmark ───────────────────────────────────────────────────

    def _crop_face(self, frame: np.ndarray, box: np.ndarray) -> tuple:
        """Crop face region with padding, return (crop_rgb, transform)."""
        import cv2
        h, w = frame.shape[:2]

        # Expand box by 25 % for context
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        pad = max(bw, bh) * 0.25
        x1 = max(0, int((box[0] - pad) * w))
        y1 = max(0, int((box[1] - pad) * h))
        x2 = min(w, int((box[2] + pad) * w))
        y2 = min(h, int((box[3] + pad) * h))

        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (self.LM_SIZE, self.LM_SIZE))
        inp = (resized.astype(np.float32) / 255.0)[np.newaxis]
        return inp, (x1, y1, x2 - x1, y2 - y1)

    def _decode_landmarks(
        self, raw: np.ndarray, presence: float, roi: tuple, frame_shape: tuple,
    ) -> Optional[List[_Landmark]]:
        """Decode 478 landmarks and map to normalized [0,1] image coordinates."""
        if presence < self.FACE_PRESENCE_THRESH:
            return None

        pts = raw.reshape(-1, 3)[:self.NUM_LANDMARKS]

        rx, ry, rw, rh = roi
        fh, fw = frame_shape[:2]

        landmarks = []
        for x_local, y_local, z_local in pts:
            # Local coords are in [0, LM_SIZE] range
            nx = (rx + x_local / self.LM_SIZE * rw) / fw
            ny = (ry + y_local / self.LM_SIZE * rh) / fh
            nz = z_local / self.LM_SIZE  # depth relative
            landmarks.append(_Landmark(nx, ny, nz))
        return landmarks

    # ── public ─────────────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray) -> FaceMeshResult:
        """Full pipeline: detect face → extract landmarks."""
        # Step 1: Face detection
        det_in = self._preprocess_det(frame_rgb)
        regressors, scores = self._det.run(None, {self._det_input: det_in})
        box = self._decode_detections(regressors, scores)

        if box is None:
            self._prev_box = None
            return FaceMeshResult(landmarks=None, has_face=False)

        self._prev_box = box

        # Step 2: Crop face and run landmarks
        lm_in, roi = self._crop_face(frame_rgb, box)
        outputs = self._lm.run(None, {self._lm_input: lm_in})
        raw_lm = outputs[0].flatten()
        raw_p = float(np.clip(outputs[1].flatten()[0], -80, 80))
        presence = 1.0 / (1.0 + np.exp(-raw_p))

        landmarks = self._decode_landmarks(raw_lm, presence, roi, frame_rgb.shape)
        if landmarks is None:
            return FaceMeshResult(landmarks=None, has_face=False)

        return FaceMeshResult(landmarks=landmarks, has_face=True)

    def close(self):
        del self._det
        del self._lm


# ═══════════════════════════════════════════════════════════════════
#  Download & model extraction helpers
# ═══════════════════════════════════════════════════════════════════

def _download_and_extract() -> bool:
    """Download task bundle and extract ONNX models if not present."""
    if DET_ONNX.exists() and LM_ONNX.exists():
        return True

    if not TASK_PATH.exists():
        try:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading face_landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, TASK_PATH)
        except Exception as e:
            logger.warning(f"Model download failed: {e}")
            return False

    # ONNX conversion requires tf2onnx + tensorflow
    if not DET_ONNX.exists() or not LM_ONNX.exists():
        try:
            _extract_and_convert()
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
            return False

    return DET_ONNX.exists() and LM_ONNX.exists()


def _extract_and_convert():
    """Extract TFLite from task bundle and convert to ONNX."""
    import subprocess

    with open(TASK_PATH, "rb") as f:
        data = f.read()
    pk = data.find(b"PK\x03\x04")
    if pk < 0:
        raise RuntimeError("Cannot find zip in task bundle")

    zf = zipfile.ZipFile(io.BytesIO(data[pk:]))

    tflite_names = {
        "face_detector.tflite": DET_ONNX,
        "face_landmarks_detector.tflite": LM_ONNX,
    }
    for tfl_name, onnx_path in tflite_names.items():
        if onnx_path.exists():
            continue
        tfl_path = MODEL_DIR / tfl_name
        tfl_path.write_bytes(zf.read(tfl_name))

        logger.info(f"Converting {tfl_name} -> ONNX ...")
        subprocess.run(
            [
                "python", "-m", "tf2onnx.convert",
                "--tflite", str(tfl_path),
                "--output", str(onnx_path),
                "--opset", "15",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"Created {onnx_path.name}")


# ═══════════════════════════════════════════════════════════════════
#  Unified FaceMesh class (public API)
# ═══════════════════════════════════════════════════════════════════

class FaceMesh:
    """
    GPU-accelerated Face Mesh with automatic fallback.

    Initialization order:
        1. ONNX Runtime GPU (DirectML/CUDA) — fastest on Windows
        2. MediaPipe Tasks API (CPU)
        3. MediaPipe Solutions API (CPU, legacy)

    Usage:
        mesh = FaceMesh(use_gpu=True)
        result = mesh.process(frame_rgb)
        if result.has_face:
            detector.process_frame(result.landmarks, w, h)
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = True,
    ):
        self._onnx: Optional[_FaceMeshONNX] = None
        self._mp_landmarker = None
        self._legacy_mesh = None
        self._mode = "none"
        self._frame_ts = 0

        # --- 1. Try MediaPipe Tasks API (reliable, default) ---
        if self._try_tasks_api(max_num_faces, min_detection_confidence, min_tracking_confidence):
            return

        # --- 3. Fallback: legacy Solutions API ---
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        self._legacy_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mode = "legacy"
        self.backend = "cpu (legacy)"
        logger.info("Face Mesh: Solutions API (CPU)")

    def _try_tasks_api(self, max_faces, det_conf, track_conf) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            if not TASK_PATH.exists():
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, TASK_PATH)

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(TASK_PATH),
                    delegate=BaseOptions.Delegate.CPU,
                ),
                running_mode=RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=det_conf,
                min_face_presence_confidence=track_conf,
                min_tracking_confidence=track_conf,
            )
            self._mp_landmarker = FaceLandmarker.create_from_options(options)
            self._mode = "tasks"
            self.backend = "cpu (tasks)"
            logger.info("Face Mesh: Tasks API (CPU)")
            return True
        except Exception as e:
            logger.warning(f"Tasks API failed: {e}")
            return False

    # ── Processing ─────────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray) -> FaceMeshResult:
        if self._mode == "onnx":
            return self._onnx.process(frame_rgb)
        elif self._mode == "tasks":
            return self._process_tasks(frame_rgb)
        else:
            return self._process_legacy(frame_rgb)

    def _process_tasks(self, frame_rgb: np.ndarray) -> FaceMeshResult:
        import mediapipe as mp
        self._frame_ts += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._mp_landmarker.detect_for_video(mp_image, self._frame_ts)
        if result.face_landmarks:
            return FaceMeshResult(landmarks=result.face_landmarks[0], has_face=True)
        return FaceMeshResult(landmarks=None, has_face=False)

    def _process_legacy(self, frame_rgb: np.ndarray) -> FaceMeshResult:
        results = self._legacy_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return FaceMeshResult(
                landmarks=results.multi_face_landmarks[0].landmark,
                has_face=True,
            )
        return FaceMeshResult(landmarks=None, has_face=False)

    def close(self):
        if self._onnx:
            self._onnx.close()
        if self._mp_landmarker:
            self._mp_landmarker.close()
        if self._legacy_mesh:
            self._legacy_mesh.close()
