import dlib
import cv2
import numpy

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


data_folder_deidentification = "/media/matej/D/Diplomski/baza/rezultati/deidentification_destination_all_images_all_k_no_comparing_landmar_distances_spec"

if __name__ == "__main__":
    image_name = "/media/matej/D/Diplomski/baza/rezultati/deidentification_destination_all_images_all_k_no_comparing_landmar_distances_spec/001/001_k_5.ppm"
    detector = FacialLandmarkDetector(image_name)
    image = detector.getImage()

    parts = detector.detectFacialLandmarks(draw = False, normalize=False, numpy_format = True)
    
    [[max_x,max_y]] = parts.max(axis=0).tolist()
    [[min_x,min_y]] = parts.min(axis=0).tolist()
    parts = numpy.vstack([parts, [(int)((max_x+min_x)/2), min_y-10]])
    parts = numpy.vstack([parts, [(int)(((max_x+min_x)/2 + min_x)/2), min_y-10]])
    parts = numpy.vstack([parts, [(int)(((max_x+min_x)/2 + max_x)/2), min_y-10]])
    
    mask = get_face_mask(image, parts)
    
    image = (image * mask).astype(numpy.uint8)
    image = image[min_y - 20:max_y + 10, min_x - 10:max_x + 10]
    cv2.imshow("bez", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
