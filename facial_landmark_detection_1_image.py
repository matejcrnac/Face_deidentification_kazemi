import dlib
import cv2
import numpy

from FacialLandmarkDetection import *


if __name__ == "__main__":
    image_name = "/home/matej/Diplomski/baze/baza_XMVTS2/000/000_1_1.ppm"
    detector = FacialLandmarkDetector(image_name)
    #detector.detect_frontal_face(True)
    detector.showImage()
    #parts = detector.detectFacialLandmarks(True)
    #detector.showImage()
    #print(parts)
    #foundParts = detector.getFacialLandmarksOfFacePart(["Nose", "Mouth"], True)
    #detector.showImage()
    #print(foundParts)
    #ROI = detector.extractFacePart("EyeRegion")
    #cv2.imshow('image',ROI)
    cv2.waitKey(0)
