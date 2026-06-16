import os
import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class DetectionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detection"
    verbose_name = "Drowsiness Detection"

    def ready(self):
        if os.environ.get("RUN_MAIN") != "true":
            return
        if os.environ.get("DJANGO_NO_AUTOSTART"):
            logger.info("Local camera disabled (DJANGO_NO_AUTOSTART=true)")
            return

        # Delay engine start so Django is fully ready
        threading.Timer(2.0, self._start_engine).start()

    def _start_engine(self):
        try:
            from .models import SystemConfig
            from .engine import get_engine

            config = SystemConfig.get_active()
            engine = get_engine()
            engine.initialize(config)
            engine.start()
        except Exception as e:
            logger.error(f"Detection engine auto-start failed: {e}")
