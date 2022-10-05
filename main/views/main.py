import os
import numpy as np
import cv2 as cv

import tensorflow as tf
# from tflite_runtime import interpreter as tflite
from django.shortcuts import render

CURRENT_PATH = os.path.dirname(__file__)

def main_page(request):
    """Main page"""
    return render(request, "main.html")

def training():
    try:
        interpreter = tf.lite.Interpreter(model_path=f"{CURRENT_PATH}/model.tflite")
        interpreter.allocate_tensors()

        train = interpreter.get_signature_runner("train")
        infer = interpreter.get_signature_runner("infer")
        save = interpreter.get_signature_runner("save")

        new_data = []
        labels = [
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
        ]
        images = [
            f"{CURRENT_PATH}/datasets/ifd1.jpg",
            f"{CURRENT_PATH}/datasets/ifd2.jpg",
            f"{CURRENT_PATH}/datasets/ifd3.jpg",
            f"{CURRENT_PATH}/datasets/ifd4.jpg",
            f"{CURRENT_PATH}/datasets/ifd5.jpg",
            f"{CURRENT_PATH}/datasets/ifd6.jpg",
            f"{CURRENT_PATH}/datasets/ifd7.jpg",
            f"{CURRENT_PATH}/datasets/ifd8.jpg",
            f"{CURRENT_PATH}/datasets/ifd9.jpg",
            f"{CURRENT_PATH}/datasets/ifd10.jpg",
            f"{CURRENT_PATH}/datasets/ifd11.jpg",
            f"{CURRENT_PATH}/datasets/jegs1.jpg",
            f"{CURRENT_PATH}/datasets/jegs2.jpg",
            f"{CURRENT_PATH}/datasets/jegs3.jpg",
            f"{CURRENT_PATH}/datasets/jegs4.jpg",
            f"{CURRENT_PATH}/datasets/jegs5.jpg",
            f"{CURRENT_PATH}/datasets/jegs6.jpg",
            f"{CURRENT_PATH}/datasets/jegs7.jpg",
        ]

        for id, image in enumerate(images):
            new_data.append(preprocess_img(image))

        new_data = np.array(new_data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        for i in range(10):
            result = train(x=new_data, y=labels)
            print("------------------------")
            print(f"EPOCH: {i}: Loss: {result['loss']}")
            print("------------------------")

        save(checkpoint_path=np.array(f"{CURRENT_PATH}/model.ckpt", dtype=np.string_))
    except Exception as exc:
        print("======================")
        print("ERRORRR")
        print("======================")
        print(exc)

def detect_face():
    try:
        interpreter = tf.lite.Interpreter(model_path=f"{CURRENT_PATH}/model.tflite")
        interpreter.allocate_tensors()
        
        infer = interpreter.get_signature_runner("infer")
        restore = interpreter.get_signature_runner("restore")
        
        if os.path.isfile(f"{CURRENT_PATH}/model.ckpt"):
            restore(checkpoint_path=np.array(f"{CURRENT_PATH}/model.ckpt", dtype=np.string_))

        labels = get_labels()

        img1 = preprocess_img(f"{CURRENT_PATH}/datasets/ifd-test.jpg")
        img2 = preprocess_img(f"{CURRENT_PATH}/datasets/jegs-test.jpg")
        img3 = preprocess_img(f"{CURRENT_PATH}/datasets/jrg-test.jpg")
        result = infer(x=np.array([img1, img2, img3], dtype=np.float32))

        print("+++++++++++++++++++++++++++++")
        print("Before the training:")
        
        index = np.argmax(result["logits"][0])
        print(result['output'])
        # print(f"Prob: {result['output'][0][index]}")
        # print(labels[np.argmax(result["logits"][0])])
        print("+++++++++++++++++++++++++++++")
    except Exception as exc:
        print(exc)

def get_labels() -> list:
    labels = [
        "ifd", "rdc", "3", "4",
        "5", "6", "7", "8",
        "9", "10", "11", "12",
        "13", "14", "15", "16",
        "17", "18", "19", "20",
    ]

    return labels

def preprocess_img(img_path:str) -> np.numarray:
    try:
        IMG_SIZE = 224

        img = cv.imread(img_path)
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        # img = img / 255.0
        img = np.float32(img)

        return img
    except Exception as exc:
        print(exc)

# training()
# detect_face()