import dlib
import cv2
import numpy

from FacialLandmarkDetection import *


if __name__ == "__main__":
    image_name = "/media/matej/D/Databases/baze/helen-master/helen-master/data/img/106242334_1.jpg"
    detector = FacialLandmarkDetector(image_name)
    #detector.detect_frontal_face(True)
    #detector.showImage()
    parts = detector.detectFacialLandmarks(draw=True, normalize=False)
    detector.showImage()
    #print(parts)
    #foundParts = detector.getFacialLandmarksOfFacePart(["Nose", "Mouth"], True)
    #detector.showImage()
    #print(foundParts)
    #ROI = detector.extractFacePart("EyeRegion")
    #cv2.imshow('image',ROI)
    cv2.waitKey(0)
