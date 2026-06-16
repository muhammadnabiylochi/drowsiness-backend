from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/status", views.api_status, name="api-status"),
    path("api/stats", views.api_stats, name="api-stats"),
    path("api/config", views.api_config, name="api-config"),
    path("api/ear-history", views.api_ear_history, name="api-ear-history"),
    path("api/video-feed", views.video_feed, name="api-video-feed"),
    path("api/sessions", views.api_sessions, name="api-sessions"),
    path("api/events", views.api_events, name="api-events"),
]
