import signal
import sys
from django.core.management.base import BaseCommand
from detection.models import SystemConfig
from detection.engine import get_engine


class Command(BaseCommand):
    help = "Start the drowsiness detection engine manually"

    def handle(self, *args, **options):
        config = SystemConfig.get_active()
        engine = get_engine()
        engine.initialize(config)
        engine.start()
        self.stdout.write(self.style.SUCCESS(
            f"Detection engine started (backend={engine.face_mesh.backend})"
        ))
        self.stdout.write("Press Ctrl+C to stop...")

        def shutdown(sig, frame):
            engine.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Keep alive
        try:
            while True:
                signal.pause()
        except AttributeError:
            # Windows doesn't have signal.pause
            import time
            while True:
                time.sleep(1)
