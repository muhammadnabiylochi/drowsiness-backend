import time
import json
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import SystemConfig, DetectionSession, DetectionEvent


def _get_engine():
    from .engine import get_engine
    return get_engine()


def dashboard(request):
    return render(request, "detection/dashboard.html")


def api_status(request):
    engine = _get_engine()
    result = engine.get_result()
    if result is None:
        return JsonResponse({"state": "no_data", "message": "No frames processed yet"})
    return JsonResponse({
        "state": result.state.value,
        "ear_left": result.ear_left,
        "ear_right": result.ear_right,
        "ear_avg": result.ear_avg,
        "mar": result.mar,
        "blink_detected": result.blink_detected,
        "total_blinks": result.total_blinks,
        "closed_frames": result.closed_frame_count,
        "yawn_frames": result.yawn_frame_count,
        "perclos": result.perclos,
        "head_roll": result.head_roll,
        "head_pitch": result.head_pitch,
        "confidence": result.confidence,
        "timestamp": result.timestamp,
    })


def api_stats(request):
    engine = _get_engine()
    if engine.detector is None:
        return JsonResponse({"error": "Detector not initialized"})
    s = engine.detector.stats
    uptime = time.time() - engine.server_start_time
    return JsonResponse({
        "total_frames": s.total_frames,
        "total_blinks": s.total_blinks,
        "drowsy_events": s.drowsy_events,
        "yawn_events": s.yawn_events,
        "avg_ear": round(s.avg_ear, 4),
        "avg_mar": round(s.avg_mar, 4),
        "perclos": round(s.perclos, 4),
        "uptime_seconds": round(uptime, 1),
        "fps": round(s.total_frames / max(uptime, 0.001), 1),
        "alerts_triggered": engine.alert_manager.alert_count if engine.alert_manager else 0,
    })


@csrf_exempt
@require_http_methods(["GET", "PUT"])
def api_config(request):
    engine = _get_engine()
    if request.method == "GET":
        if engine.detector is None:
            return JsonResponse({"error": "Detector not initialized"})
        return JsonResponse({
            "ear_threshold": engine.detector.ear_threshold,
            "mar_threshold": engine.detector.mar_threshold,
            "drowsy_frame_threshold": engine.detector.drowsy_frame_threshold,
            "yawn_frame_threshold": engine.detector.yawn_frame_threshold,
            "max_blink_frames": engine.detector.max_blink_frames,
            "perclos_window_sec": engine.detector.perclos_window_sec,
            "perclos_threshold": engine.detector.perclos_threshold,
        })
    data = json.loads(request.body)
    engine.update_thresholds(**data)
    config = SystemConfig.get_active()
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.save()
    return JsonResponse({"status": "ok", "updated": data})


def api_ear_history(request):
    engine = _get_engine()
    d = engine.detector
    return JsonResponse({
        "ear": d.get_ear_history() if d else [],
        "mar": d.get_mar_history() if d else [],
        "ear_threshold": d.ear_threshold if d else 0.22,
        "mar_threshold": d.mar_threshold if d else 0.55,
    })


async def _generate_mjpeg():
    import cv2
    import asyncio
    engine = _get_engine()
    while True:
        if engine.camera:
            ret, frame = engine.camera.read()
            if ret and frame is not None:
                result = engine.get_result()
                if result is not None:
                    state_key = result.state.value
                    state_labels = {
                        "awake": "HUSHYOR",
                        "drowsy": "UYQULI",
                        "yawning": "ESNAMOQDA",
                        "drowsy_yawning": "UYQULI + ESNAMOQDA",
                        "no_face": "YUZ TOPILMADI",
                        "falling_forward": "BOSH OLDINGA TUSHMOQDA",
                        "falling_back": "BOSH ORQAGA TUSHMOQDA",
                        "falling_left": "BOSH CHAPGA TUSHMOQDA",
                        "falling_right": "BOSH O'NGGA TUSHMOQDA",
                    }
                    state_text = state_labels.get(state_key, state_key.upper().replace("_", " "))
                    color = (0, 255, 0) if state_key == "awake" else (0, 0, 255)
                    cv2.putText(frame, state_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
        await asyncio.sleep(1 / 25)


async def video_feed(request):
    return StreamingHttpResponse(
        _generate_mjpeg(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def api_sessions(request):
    sessions = DetectionSession.objects.all()[:20]
    data = [{
        "id": str(s.id),
        "started_at": s.started_at.isoformat(),
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "is_active": s.is_active,
        "total_frames": s.total_frames,
        "drowsy_events": s.drowsy_events,
    } for s in sessions]
    return JsonResponse({"sessions": data})


def api_events(request):
    events = DetectionEvent.objects.all()[:50]
    data = [{
        "timestamp": e.timestamp.isoformat(),
        "state": e.state,
        "ear_avg": e.ear_avg,
        "mar": e.mar,
        "perclos": e.perclos,
        "confidence": e.confidence,
    } for e in events]
    return JsonResponse({"events": data})
