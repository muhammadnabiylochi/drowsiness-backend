# Real-Time Drowsiness Detection System

**Based on:** Safarov, F.; Akhmedov, F.; Abdusalomov, A.B.; Nasimov, R.; Cho, Y.I. (2023)
*"Real-Time Deep Learning-Based Drowsiness Detection: Leveraging Computer-Vision and Eye-Blink Analyses for Enhanced Road Safety"*
Sensors 23(14), 6459. [DOI: 10.3390/s23146459](https://doi.org/10.3390/s23146459)

## Features

- Real-time face mesh detection (478 landmarks) via MediaPipe / ONNX Runtime
- EAR (Eye Aspect Ratio) based drowsiness detection (Eq. 4)
- MAR (Mouth Aspect Ratio) based yawning detection
- PERCLOS calculation (> 0.4 = high risk)
- Head pose estimation (falling left/right/back)
- GPU acceleration via ONNX Runtime DirectML (NVIDIA/AMD/Intel)
- Django admin panel for system configuration
- Live dashboard with real-time WebSocket streaming
- REST API for integration
- Alert system (sound, webhooks, logging)

## Detection Classes

| State | Accuracy | Description |
|-------|----------|-------------|
| Drowsy (eyes closed) | 95.8% | EAR below threshold for consecutive frames |
| Open eyes (awake) | 97.0% | Normal EAR values |
| Yawning | 84.0% | MAR above threshold |
| Falling right | 98.0% | Head roll > 35 degrees |
| Falling left | 100% | Head roll < -35 degrees |
| Falling back | - | Head pitch > 25 degrees |

## Architecture

```
Camera Input (Webcam / RTSP / Video)
        |
        v
Face Mesh (ONNX GPU / MediaPipe CPU)
  478 facial landmarks
        |
        v
DrowsinessDetector (core/detector.py)
  EAR = (||P2-P6|| + ||P3-P5||) / (2*||P1-P4||)   [Eq. 4]
  S(k) = 1 if |w(k)| >= T_S, else 0                [Eq. 5]
  PERCLOS, MAR, Head Pose
        |
        v
Django Web Server
  Admin Panel (/admin/)
  Dashboard (/)
  REST API (/api/*)
  WebSocket (/ws/stream)
  SQLite Database
```

## Project Structure

```
drowsiness-backend/
├── manage.py                    # Django entry point
├── drowsiness_project/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── asgi.py                  # ASGI (HTTP + WebSocket)
├── detection/                   # Django app
│   ├── models.py                # DB models (Config, Sessions, Events, Alerts)
│   ├── admin.py                 # Admin panel configuration
│   ├── views.py                 # REST API + dashboard views
│   ├── engine.py                # Detection loop singleton
│   ├── consumers.py             # WebSocket consumer
│   └── management/commands/
│       └── startdetection.py    # Manual engine start command
├── core/                        # Detection engine (framework-agnostic)
│   ├── detector.py              # EAR/MAR/PERCLOS calculations
│   ├── camera.py                # Camera stream (Webcam/RTSP/File)
│   ├── alert.py                 # Alert system (sound/webhook/log)
│   └── face_mesh.py             # GPU face mesh (ONNX/MediaPipe)
├── models/                      # Pre-trained ML models (.onnx)
├── templates/detection/         # Dashboard HTML template
├── static/                      # CSS, JS
├── run_standalone.py            # Standalone CLI mode (OpenCV window)
└── requirements.txt
```

## Quick Start

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd drowsiness-backend

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. GPU support (optional, recommended)

```bash
pip install onnxruntime-directml
```

### 4. Setup database

```bash
python manage.py migrate
python manage.py createsuperuser
```

### 5. Run server

```bash
python manage.py runserver 0.0.0.0:8000
```

### 6. Open in browser

- **Dashboard:** http://localhost:8000/
- **Admin Panel:** http://localhost:8000/admin/
- **API Status:** http://localhost:8000/api/status

## Admin Panel

Access at `/admin/` to manage:

- **System Configuration** -- Camera source, EAR/MAR thresholds, PERCLOS settings, alerts, GPU toggle
- **Detection Sessions** -- History of all camera sessions with stats
- **Detection Events** -- Every state transition (awake -> drowsy, etc.) with metrics
- **Alert Logs** -- All triggered alerts with timestamps

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Live dashboard |
| GET | `/api/status` | Current detection state (JSON) |
| GET | `/api/stats` | Session statistics |
| GET/PUT | `/api/config` | Read/update thresholds |
| GET | `/api/ear-history` | EAR/MAR waveform data |
| GET | `/api/video-feed` | MJPEG video stream |
| WS | `/ws/stream` | Real-time WebSocket stream |
| GET | `/api/sessions` | Session history |
| GET | `/api/events` | Event history |

## Standalone Mode (no web server)

```bash
python run_standalone.py                     # Webcam + OpenCV window
python run_standalone.py --source 0          # Explicit webcam
python run_standalone.py --headless          # No GUI (edge/server)
python run_standalone.py --no-gpu            # Force CPU mode
```

## Environment Variables

```bash
# Camera
CAM_SOURCE=0                    # 0=webcam, rtsp://..., or video file
CAM_WIDTH=640
CAM_HEIGHT=480

# Detection thresholds
DET_EAR_THRESHOLD=0.22          # EAR below this = eyes closed
DET_MAR_THRESHOLD=0.55          # MAR above this = yawning
DET_DROWSY_FRAME_THRESHOLD=15   # Consecutive frames -> DROWSY

# Django
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=true

# Alerts
ALERT_ENABLED=true
ALERT_WEBHOOK_URL=               # Optional webhook URL
```

## Key Algorithms (from paper)

**EAR -- Eye Aspect Ratio (Equation 4):**
```
EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
```

**Threshold Classification (Equation 5):**
```
S(k) = 1  if |w(k)| >= T_S   (eye closed -> drowsy candidate)
S(k) = 0  if |w(k)| < T_S    (eye open -> awake)
```

**PERCLOS:**
```
PERCLOS = (frames with eyes closed) / (total frames in window)
PERCLOS > 0.4 -> high drowsiness risk
```

## Hardware Requirements

- Camera: Webcam, IP camera (RTSP), or video file
- Camera distance: 0.3m - 0.6m from driver
- GPU (optional): NVIDIA/AMD/Intel for ONNX Runtime DirectML acceleration
- Tested on: NVIDIA RTX 4500 Ada (24GB)

## Tech Stack

- **Backend:** Django 5.x + Daphne (ASGI)
- **Real-time:** Django Channels (WebSocket)
- **CV/ML:** MediaPipe, OpenCV, ONNX Runtime
- **Database:** SQLite (default), PostgreSQL (production)
- **GPU:** ONNX Runtime DirectML / CUDA

## License

Based on the research paper by Safarov et al. (2023), published under CC BY 4.0.
