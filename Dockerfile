FROM python:3.12-slim

# OpenCV va MediaPipe uchun system kutubxonalar
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libgles2 libegl1 libegl-mesa0 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*

ENV LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD python manage.py migrate --run-syncdb && \
    daphne -b 0.0.0.0 -p 8000 drowsiness_project.asgi:application
