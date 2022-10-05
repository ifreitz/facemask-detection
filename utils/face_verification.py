import os
import urllib.request as ur
import face_recognition

from deepface import DeepFace
from deepface.detectors import FaceDetector

CURRENT_PATH = os.path.dirname(__file__)
THRESH_HOLD = 0.75

def get_known_images() -> dict:
    """Return list of known images"""

    known_images = {
        "JEGS": [
            f"{CURRENT_PATH}/face_data/jegs1.jpg",
            f"{CURRENT_PATH}/face_data/jegs2.jpg",
        ],
        "RDC": [
            f"{CURRENT_PATH}/face_data/rdc1.jpg",
            f"{CURRENT_PATH}/face_data/rdc2.jpg",
            f"{CURRENT_PATH}/face_data/rdc3.jpg",
        ],
        "IFD": [
            f"{CURRENT_PATH}/face_data/ifd1.jpg",
            f"{CURRENT_PATH}/face_data/ifd2.jpg",
            f"{CURRENT_PATH}/face_data/ifd3.jpg",
            f"{CURRENT_PATH}/face_data/ifd4.jpg",
        ],
    }

    return known_images

def load_base64_img(base64img):
    """Load base64 image"""

    decoded_image = ur.urlopen(base64img)
    unknown_image = face_recognition.load_image_file(decoded_image)
    return unknown_image

def compare_faces(unknown_image):
    """Compare faces in the provided images to the known images"""

    compare_result = []

    # Returns A list of tuples of found face locations in css (top, right, bottom, left) order
    face_locations = face_recognition.face_locations(unknown_image)

    if len(face_locations) > 0:
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        
        for location in face_locations:
            compare_result.append({
                "face_location": location,
                "verified": False,
                "verified_person": "Unknown",
                "score": 0
            })

        known_images = get_known_images()

        for idx, unknown_encoding in enumerate(unknown_encodings):
            for known_img in known_images:
                verified = 0
                num_images = len(known_images[known_img])
                imgs = []

                for img in known_images[known_img]:
                    known_image = face_recognition.load_image_file(img)
                    imgs.append(face_recognition.face_encodings(known_image)[0])
                    
                results = face_recognition.compare_faces(imgs, unknown_encoding, tolerance=0.5)

                for result in results:
                    if result:
                        verified += 1
                        
                if (verified / num_images) >= THRESH_HOLD:
                    compare_result[idx]["verified"] = True
                    compare_result[idx]["verified_person"] = known_img
                    compare_result[idx]["score"] = (verified / num_images)

                    break

    return compare_result


def verify_face(img) -> dict:
    """Verify face"""

    result = DeepFace.verify(img, f"{CURRENT_PATH}/face_data/ifd1.jpg")
    
    detector = FaceDetector.build_model("opencv")
    face = FaceDetector.detect_faces(detector, "opencv", img)

    # print(result)
    print(len(face))

    return result