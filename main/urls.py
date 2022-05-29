from django.urls import path

from main.views import *

urlpatterns = [
    path("", main_page, name="landing_page"),
]