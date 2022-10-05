from utils.request import *
from utils.face_verification import *
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# return HttpResponse("Saved successfully!", status=200)
# return JsonResponse(settings.get_dictionary_value(), safe=True, status=200)

def facial_recognition_page(request):
    """Facial recognition page"""
    return render(request, "facial_recognition.html")

def facial_recognition_page_v2(request):
    """Facial recognition page version 2"""
    return render(request, "facial_recognition_v2.html")

def verify(request):
    """Verify person in the photo"""

    data = extract_json_data(request)
    # unknown_image = load_base64_img(data["photo"])
    # results = compare_faces(unknown_image)
    results = verify_face(data["photo"])

    return JsonResponse(results, safe=False, status=200)