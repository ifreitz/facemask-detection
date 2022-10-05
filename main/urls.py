from django.urls import path

from main.views import main, facial_recognition

urlpatterns = [
    path("", main.main_page, name="landing_page"),
    path("facial-recognition/", facial_recognition.facial_recognition_page),
    path("facial-recognition/v2/", facial_recognition.facial_recognition_page_v2),
    path("facial-recognition/verify/", facial_recognition.verify),
]