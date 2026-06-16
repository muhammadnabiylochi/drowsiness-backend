import uuid
from django.db import models


class SystemConfig(models.Model):
    """Singleton database-backed configuration. Editable from admin panel."""

    name = models.CharField(max_length=100, default="default", unique=True)

    # Camera
    camera_source = models.CharField(max_length=500, default="0",
        help_text="0=webcam, rtsp://... , or video file path")
    camera_width = models.IntegerField(default=640)
    camera_height = models.IntegerField(default=480)
    camera_fps = models.IntegerField(default=30)
    camera_flip_horizontal = models.BooleanField(default=True)

    # Detection thresholds (Safarov et al. 2023)
    ear_threshold = models.FloatField(default=0.22,
        help_text="EAR below this = eyes closed (Eq. 4)")
    mar_threshold = models.FloatField(default=0.55,
        help_text="MAR above this = yawning (Section 2.5)")
    drowsy_frame_threshold = models.IntegerField(default=15,
        help_text="Consecutive closed-eye frames -> DROWSY")
    yawn_frame_threshold = models.IntegerField(default=10,
        help_text="Consecutive yawning frames -> YAWNING")
    max_blink_frames = models.IntegerField(default=8,
        help_text="Max frames for a normal blink")

    # PERCLOS
    perclos_window_sec = models.FloatField(default=60.0,
        help_text="Time window (seconds) for PERCLOS")
    perclos_threshold = models.FloatField(default=0.4,
        help_text="PERCLOS > 0.4 = high drowsiness risk")

    # Alerts
    alert_enabled = models.BooleanField(default=True)
    alert_sound_file = models.CharField(max_length=500, blank=True, default="")
    alert_cooldown_sec = models.FloatField(default=5.0)
    alert_webhook_url = models.URLField(blank=True, default="")

    # GPU (ONNX DirectML — experimental, landmark mapping needs calibration)
    gpu_enabled = models.BooleanField(default=False)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Configuration"
        verbose_name_plural = "System Configuration"

    def __str__(self):
        return f"Config: {self.name}"

    @classmethod
    def get_active(cls):
        obj, _ = cls.objects.get_or_create(name="default")
        return obj


class DetectionSession(models.Model):
    """Tracks a camera session from start to stop."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    camera_source = models.CharField(max_length=500, default="0")
    is_active = models.BooleanField(default=True)

    total_frames = models.IntegerField(default=0)
    total_blinks = models.IntegerField(default=0)
    drowsy_events = models.IntegerField(default=0)
    yawn_events = models.IntegerField(default=0)
    avg_ear = models.FloatField(default=0.0)
    avg_mar = models.FloatField(default=0.0)
    avg_perclos = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        status = "Active" if self.is_active else "Ended"
        return f"Session {self.id.hex[:8]} ({status})"


class DetectionEvent(models.Model):
    """Records each state transition."""

    class State(models.TextChoices):
        AWAKE = "awake", "Awake"
        DROWSY = "drowsy", "Drowsy"
        YAWNING = "yawning", "Yawning"
        DROWSY_YAWNING = "drowsy_yawning", "Drowsy + Yawning"
        FALLING_RIGHT = "falling_right", "Falling Right"
        FALLING_LEFT = "falling_left", "Falling Left"
        FALLING_BACK = "falling_back", "Falling Back"
        NO_FACE = "no_face", "No Face"

    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, related_name="events")
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    state = models.CharField(max_length=20, choices=State.choices)
    previous_state = models.CharField(max_length=20, choices=State.choices, blank=True)
    ear_left = models.FloatField(default=0)
    ear_right = models.FloatField(default=0)
    ear_avg = models.FloatField(default=0)
    mar = models.FloatField(default=0)
    perclos = models.FloatField(default=0)
    head_roll = models.FloatField(default=0)
    head_pitch = models.FloatField(default=0)
    confidence = models.FloatField(default=0)
    total_blinks = models.IntegerField(default=0)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["session", "state"]),
        ]

    def __str__(self):
        return f"{self.timestamp:%H:%M:%S} {self.get_state_display()}"


class AlertLog(models.Model):
    """Records triggered alerts."""

    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE, related_name="alerts")
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    state = models.CharField(max_length=20, choices=DetectionEvent.State.choices)
    ear_avg = models.FloatField(default=0)
    mar = models.FloatField(default=0)
    perclos = models.FloatField(default=0)
    total_blinks = models.IntegerField(default=0)
    head_roll = models.FloatField(default=0)
    confidence = models.FloatField(default=0)
    alert_number = models.IntegerField(default=0)
    webhook_sent = models.BooleanField(default=False)
    sound_played = models.BooleanField(default=False)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"Alert #{self.alert_number} - {self.state}"
