"""
Alert System
=============
Handles drowsiness alerts: sound alarms, logging, webhooks.
"""

import time
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Multi-channel alerting for drowsiness events.
    
    Channels:
        1. Sound alarm (local speaker / buzzer)
        2. File logging (CSV format for analysis)
        3. Webhook (HTTP POST to fleet management / monitoring)
    """

    def __init__(
        self,
        enabled: bool = True,
        sound_file: Optional[str] = None,
        cooldown_sec: float = 5.0,
        log_file: str = "logs/drowsiness_events.log",
        webhook_url: Optional[str] = None,
    ):
        self.enabled = enabled
        self.sound_file = sound_file
        self.cooldown_sec = cooldown_sec
        self.log_file = Path(log_file)
        self.webhook_url = webhook_url
        self._last_alert_time = 0.0
        self._alert_count = 0

        # Ensure log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV header if new file
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                f.write("timestamp,state,ear_avg,mar,perclos,blinks,head_roll,confidence\n")

        logger.info(f"AlertManager initialized: log={self.log_file}, cooldown={cooldown_sec}s")

    def trigger(
        self,
        state: str,
        ear_avg: float = 0.0,
        mar: float = 0.0,
        perclos: float = 0.0,
        blinks: int = 0,
        head_roll: float = 0.0,
        confidence: float = 0.0,
    ):
        """
        Trigger an alert for a drowsiness event.
        Respects cooldown to avoid alert fatigue.
        """
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_alert_time < self.cooldown_sec:
            return

        self._last_alert_time = now
        self._alert_count += 1

        # 1. Log to file
        self._log_event(state, ear_avg, mar, perclos, blinks, head_roll, confidence)

        # 2. Sound alarm (non-blocking)
        if state in ("drowsy", "drowsy_yawning", "falling_right", "falling_left", "falling_back"):
            threading.Thread(target=self._play_alarm, daemon=True).start()

        # 3. Webhook
        if self.webhook_url:
            threading.Thread(
                target=self._send_webhook,
                args=(state, ear_avg, mar, perclos, confidence),
                daemon=True,
            ).start()

        logger.warning(f"ALERT #{self._alert_count}: {state} (EAR={ear_avg:.3f}, MAR={mar:.3f})")

    def _log_event(self, state, ear_avg, mar, perclos, blinks, head_roll, confidence):
        """Append event to CSV log."""
        try:
            ts = datetime.now().isoformat()
            with open(self.log_file, "a") as f:
                f.write(f"{ts},{state},{ear_avg:.4f},{mar:.4f},{perclos:.4f},{blinks},{head_roll:.2f},{confidence:.3f}\n")
        except Exception as e:
            logger.error(f"Log write error: {e}")

    def _play_alarm(self):
        """Play alarm sound."""
        try:
            if self.sound_file and Path(self.sound_file).exists():
                # Try platform-appropriate playback
                try:
                    # Linux (Jetson / Ubuntu)
                    subprocess.run(
                        ["aplay", "-q", self.sound_file],
                        timeout=5,
                        capture_output=True,
                    )
                except FileNotFoundError:
                    try:
                        # macOS
                        subprocess.run(
                            ["afplay", self.sound_file],
                            timeout=5,
                            capture_output=True,
                        )
                    except FileNotFoundError:
                        logger.debug("No audio player found")
            else:
                # Generate beep via system bell
                try:
                    # Console beep (works on most Linux/Jetson)
                    subprocess.run(
                        ["beep", "-f", "1000", "-l", "500"],
                        timeout=3,
                        capture_output=True,
                    )
                except FileNotFoundError:
                    print("\a", end="", flush=True)  # Terminal bell
        except Exception as e:
            logger.debug(f"Sound playback error: {e}")

    def _send_webhook(self, state, ear_avg, mar, perclos, confidence):
        """Send alert to external webhook."""
        try:
            import urllib.request

            payload = json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event": "drowsiness_alert",
                "state": state,
                "metrics": {
                    "ear_avg": ear_avg,
                    "mar": mar,
                    "perclos": perclos,
                    "confidence": confidence,
                },
            }).encode()

            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Webhook sent to {self.webhook_url}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    @property
    def alert_count(self) -> int:
        return self._alert_count
