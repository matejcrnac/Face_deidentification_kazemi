import dlib
import cv2
import numpy
import os

from FacialLandmarkDetection import *

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

FEATHER_AMOUNT = 3

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    draw_convex_hull(im,
                     landmarks,
                     color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
def get_images_Path(folder, extension):
    images_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                fName = os.path.join(root, file)
                images_paths.append(fName)

    return sorted(images_paths)


data_folder_deidentification = "/home/matej/Diplomski/baze/deidentification_database/baza_deidentification_Images"

destination_folder = "/media/matej/D/Diplomski/baza/rezultati"
if __name__ == "__main__":
    
    images_paths = get_images_Path(data_folder_deidentification, ".ppm")
    
    destination = destination_folder + "/original_extracted" + "/"
    if not os.path.exists(destination):
        os.makedirs(destination)

    for image_name in images_paths:
        detector = FacialLandmarkDetector(image_name)
        #detector.detect_frontal_face(True)
        #detector.showImage()
        image = detector.getImage()

        parts = detector.detectFacialLandmarks(draw = False, normalize=False, numpy_format = True)
        
        [[max_x,max_y]] = parts.max(axis=0).tolist()
        [[min_x,min_y]] = parts.min(axis=0).tolist()
        parts = numpy.vstack([parts, [(int)((max_x+min_x)/2), min_y-10]])
        parts = numpy.vstack([parts, [(int)(((max_x+min_x)/2 + min_x)/2), min_y-10]])
        parts = numpy.vstack([parts, [(int)(((max_x+min_x)/2 + max_x)/2), min_y-10]])
        #parts = numpy.vstack([parts, [(int)((max_x+min_x)/2 + (max_x+min_x)/4), min_y-10]])
        #parts = numpy.vstack([parts, [(int)((max_x+min_x)/2 + (max_x+min_x)/3), min_y-10]])
        
        mask = get_face_mask(image, parts)
        
        image = (image * mask).astype(numpy.uint8)
        #cv2.imshow("bez", image)
        #cv2.waitKey(0)
        image = image[min_y - 20:max_y + 10, min_x - 10:max_x + 10]
        
        name = image_name.split("/")[-1].split("_")[0]
        destination_new = destination + name + "/"
        if not os.path.exists(destination_new):
            os.makedirs(destination_new)
        imname = destination_new + name + ".ppm"
        
        cv2.imwrite(imname, image)
        
        print("Image: " + name)
        del detector
