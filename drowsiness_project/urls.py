from django.contrib import admin
from django.urls import path, include

admin.site.site_header = "Drowsiness Detection System"
admin.site.site_title = "DDS Admin"
admin.site.index_title = "Administration"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("detection.urls")),
]
