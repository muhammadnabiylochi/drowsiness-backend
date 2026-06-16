from django.contrib import admin
from django.utils.html import format_html
from .models import SystemConfig, DetectionSession, DetectionEvent, AlertLog


@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    fieldsets = (
        ("Camera", {
            "fields": ("camera_source", "camera_width", "camera_height",
                       "camera_fps", "camera_flip_horizontal"),
        }),
        ("Detection Thresholds (Safarov et al. 2023, Eq. 4-5)", {
            "fields": ("ear_threshold", "mar_threshold",
                       "drowsy_frame_threshold", "yawn_frame_threshold",
                       "max_blink_frames"),
        }),
        ("PERCLOS", {
            "fields": ("perclos_window_sec", "perclos_threshold"),
        }),
        ("Alerts", {
            "fields": ("alert_enabled", "alert_sound_file",
                       "alert_cooldown_sec", "alert_webhook_url"),
        }),
        ("GPU", {
            "fields": ("gpu_enabled",),
        }),
    )
    list_display = ("name", "ear_threshold", "mar_threshold",
                    "drowsy_frame_threshold", "alert_enabled", "gpu_enabled",
                    "updated_at")

    def has_add_permission(self, request):
        return not SystemConfig.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(DetectionSession)
class DetectionSessionAdmin(admin.ModelAdmin):
    list_display = ("short_id", "started_at", "ended_at", "status_display",
                    "camera_source", "total_frames", "total_blinks",
                    "drowsy_events", "yawn_events", "duration_display")
    list_filter = ("is_active", "started_at")
    readonly_fields = ("id", "started_at", "ended_at", "total_frames",
                       "total_blinks", "drowsy_events", "yawn_events",
                       "avg_ear", "avg_mar", "avg_perclos")

    def short_id(self, obj):
        return obj.id.hex[:8]
    short_id.short_description = "ID"

    def status_display(self, obj):
        if obj.is_active:
            return format_html('<span style="color:green;font-weight:bold;">Running</span>')
        return format_html('<span style="color:gray;">Ended</span>')
    status_display.short_description = "Status"

    def duration_display(self, obj):
        if obj.ended_at and obj.started_at:
            delta = obj.ended_at - obj.started_at
            return f"{delta.total_seconds() / 60:.1f} min"
        return "-"
    duration_display.short_description = "Duration"


@admin.register(DetectionEvent)
class DetectionEventAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "state_colored", "ear_avg", "mar",
                    "perclos", "head_roll", "confidence", "total_blinks")
    list_filter = ("state", "session", "timestamp")
    readonly_fields = ("session", "timestamp", "state", "previous_state",
                       "ear_left", "ear_right", "ear_avg", "mar",
                       "perclos", "head_roll", "head_pitch", "confidence",
                       "total_blinks")
    date_hierarchy = "timestamp"
    list_per_page = 50

    STATE_COLORS = {
        "awake": "green", "drowsy": "red", "yawning": "orange",
        "drowsy_yawning": "red", "falling_right": "red",
        "falling_left": "red", "falling_back": "red", "no_face": "gray",
    }

    def state_colored(self, obj):
        color = self.STATE_COLORS.get(obj.state, "black")
        return format_html(
            '<span style="color:{};font-weight:bold;">{}</span>',
            color, obj.get_state_display()
        )
    state_colored.short_description = "State"

    def has_add_permission(self, request):
        return False


@admin.register(AlertLog)
class AlertLogAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "state", "alert_number", "ear_avg",
                    "mar", "perclos", "confidence")
    list_filter = ("state", "timestamp")
    readonly_fields = ("session", "timestamp", "state", "ear_avg", "mar",
                       "perclos", "total_blinks", "head_roll", "confidence",
                       "alert_number", "webhook_sent", "sound_played")
    date_hierarchy = "timestamp"

    def has_add_permission(self, request):
        return False
