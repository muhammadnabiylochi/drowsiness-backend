"""
Drowsiness Detection System — Configuration
=============================================
All thresholds based on Safarov et al. (2023), Sensors 23(14), 6459.

EAR formula (Eq.4):
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

Drowsiness classification:
    1. Yawning-based (MAR threshold)
    2. Eye-blinking-based (EAR threshold + consecutive frames)
    3. Joint yawning + eye-blinking
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class CameraConfig(BaseSettings):
    """Camera source configuration.
    
    Supports:
        - Local webcam: source = 0, 1, ...
        - IP camera (Axis P1367 etc): source = "rtsp://user:pass@192.168.1.100/axis-media/media.amp"
        - Video file: source = "/path/to/video.mp4"
    """
    source: str = Field(default="0", description="Camera source: device index, RTSP URL, or file path")
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_horizontal: bool = True  # Mirror for front-facing camera

    class Config:
        env_prefix = "CAM_"


class DetectionConfig(BaseSettings):
    """Detection thresholds from the paper.
    
    EAR threshold: ~0.22 (paper uses waveform threshold at point 34 
                   normalized between 20-45 range)
    Camera distance: 0.3m - 0.6m (paper Section 3)
    """
    # Eye Aspect Ratio (EAR) — Eq. 4
    ear_threshold: float = Field(
        default=0.22,
        description="EAR below this = eyes closed. Paper: threshold line at 34 (normalized)"
    )
    
    # Mouth Aspect Ratio (MAR) — yawning detection (Section 2.5)
    mar_threshold: float = Field(
        default=0.55,
        description="MAR above this = yawning. Based on mouth width-to-height ratio"
    )
    
    # Consecutive frames to confirm drowsiness
    drowsy_frame_threshold: int = Field(
        default=15,
        description="Number of consecutive closed-eye frames to trigger DROWSY alert"
    )
    
    # Consecutive yawning frames
    yawn_frame_threshold: int = Field(
        default=10,
        description="Number of consecutive yawning frames to trigger YAWNING alert"
    )
    
    # Blink duration limits (normal blink: 100-400ms)
    max_blink_frames: int = Field(
        default=8,
        description="Max frames for a normal blink (not drowsy)"
    )
    
    # PERCLOS: percentage of eye closure over time window
    perclos_window_sec: float = Field(
        default=60.0,
        description="Time window (seconds) for PERCLOS calculation"
    )
    perclos_threshold: float = Field(
        default=0.4,
        description="PERCLOS > 0.4 = high drowsiness risk"
    )

    class Config:
        env_prefix = "DET_"


class AlertConfig(BaseSettings):
    """Alert system configuration."""
    enabled: bool = True
    sound_file: Optional[str] = None  # Path to alert .wav file
    cooldown_sec: float = 5.0  # Minimum seconds between alerts
    log_file: str = "logs/drowsiness_events.log"
    
    # Webhook for external integration (e.g., fleet management)
    webhook_url: Optional[str] = None
    
    class Config:
        env_prefix = "ALERT_"


class ServerConfig(BaseSettings):
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    class Config:
        env_prefix = "SERVER_"


class AppConfig(BaseSettings):
    """Root configuration."""
    camera: CameraConfig = CameraConfig()
    detection: DetectionConfig = DetectionConfig()
    alert: AlertConfig = AlertConfig()
    server: ServerConfig = ServerConfig()
    gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration for face mesh")

    class Config:
        env_prefix = ""
