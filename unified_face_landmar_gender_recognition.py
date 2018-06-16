#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:02:40 2017

@author: dhingratul
"""

import imutils
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib


class UnifiedGenderDetection:
    def __init__(self, model_path):
        self.clf1 = joblib.load(model_path)  # DL Model

    def featureExtract(self, img):
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        facerec = dlib.face_recognition_model_v1(
                'gender_models/dlib_face_recognition_resnet_model_v1.dat')
        dets = detector(img, 1)
        # Now process each face we found.
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            return face_descriptor
            
    def predict(self, image_path):
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=500)

        # Male/ Female Prediction
        features = self.featureExtract(image)
        features = np.reshape(features, (1, -1))
        out = self.clf1.predict(features)
        
        #switch values -> original: 0 -> woman, 1- man ---> 0->man, 1-> woman
        if out == 0:    #woman
            out = 1
        else:   #man
            out = 0
        return out
        
    def predict_label(self, image_path):
        predicted = self.predict(image_path)
        if (predicted == 0):
            return "man"
        else:
            return "woman"

        
if __name__ == "__main__":

    imagePath_same1 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/000/000_1_1.ppm" # sve dobro,
    imagePath_same2 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/001/001_1_1.ppm" # sve dobro
    imagePath_same3 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/002/002_1_1.ppm" # spec bolji (k4 prob)
    imagePath_same4 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/004/004_1_1.ppm" # spec bolji od all dist
    imagePath_same5 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/005/005_1_1.ppm" # spec najbolji
    imagePath_same6 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/006/006_1_1.ppm"
    imagePath_same7 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/007/007_1_1.ppm" # nije dobar
    imagePath_same8 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/008/008_1_1.ppm"
    imagePath_same9 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/009/009_1_1.ppm"
    imagePath_same10 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/010/010_1_1.ppm"


    imagePath = imagePath_same3

    model_path = "gender_models/svm_model.pkl"
    model = UnifiedGenderDetection(model_path)

    predicted = model.predict(imagePath)

    print(predicted)

    label = model.predict_label(imagePath)

    print(label)
