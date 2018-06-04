import cv2
import numpy

from FacialLandmarkDetection import *


SCALE_FACTOR = 1 
FEATHER_AMOUNT = 9


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
RIGHT_UPPER_JAW = list(range(2, 3))
LEFT_UPPER_JAW = list(range(14, 15))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS )

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.

#2 groups - eye region, nose + mouth
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, 
    NOSE_POINTS +  MOUTH_POINTS,
]

#everithing one group
#OVERLAY_POINTS = [
#    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS,
#]

#OVERLAY_POINTS = [
#    NOSE_POINTS + MOUTH_POINTS,
#]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


class FaceSwap:
    def __init__(self, image_path1, image_path2):
        self.image_path1 = image_path1
        self.img1 = cv2.imread(self.image_path1, cv2.IMREAD_COLOR)
        self.image_path2 = image_path2
        self.img2 = cv2.imread(self.image_path2, cv2.IMREAD_COLOR)
        self.landmarks1 = None
        self.landmarks2 = None
        
    def load_landmarks(self, landmarks, first_image = True):
        if first_image == True:
            self.landmarks1 = landmarks
        else:
            self.landmarks2 = landmarks
        
    def swap_face(self):
        if self.landmarks1 == None:
            detector1 = FacialLandmarkDetector(self.image_path1)
            self.landmarks1 = detector1.detectFacialLandmarks(draw=False, normalize=False, numpy_format = True)
        if self.landmarks2 == None:
            detector2 = FacialLandmarkDetector(self.image_path2)
            self.landmarks2 = detector2.detectFacialLandmarks(draw=False, normalize=False, numpy_format = True)
        
        img = self.swap_algorithm()
        return img
        
        #Method calculates affine transformation matrix M
    def transformation_from_points(self, points1, points2):
        """
        Return an affine transformation [s * R | T] such that:

            sum ||s*R*p1,i + T - p2,i||^2

        is minimized.
        s - scaling
        R - rotation matrix
        T - translation vectors
        p and q are landmark points of image 1 and image 2  
        """
        # Solve the procrustes problem by subtracting centroids, scaling by the
        # standard deviation, and then using the SVD to calculate the rotation. See
        # the following for more details:
        #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

        points1 = points1.astype(numpy.float64)
        points2 = points2.astype(numpy.float64)

        #subtract mean
        c1 = numpy.mean(points1, axis=0)
        c2 = numpy.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        #divide by standard deviation
        s1 = numpy.std(points1)
        s2 = numpy.std(points2)
        points1 /= s1
        points2 /= s2

        #Use SVD (Singular Value Decomposition) to calculate rotation
        U, S, Vt = numpy.linalg.svd(points1.T * points2)

        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T

        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             numpy.matrix([0., 0., 1.])])
                             
    def warp_im(self, im, M, dshape):
        output_im = numpy.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                                  numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                                  numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                    im2_blur.astype(numpy.float64))

    def draw_convex_hull(self, im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(self, im, landmarks):
        im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

        for group in OVERLAY_POINTS:
            self.draw_convex_hull(im,
                             landmarks[group],
                             color=1)

        im = numpy.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

        return im

    def swap_algorithm(self):
        M = self.transformation_from_points(self.landmarks1[ALIGN_POINTS],self.landmarks2[ALIGN_POINTS])

        mask = self.get_face_mask(self.img1, self.landmarks2)

        warped_mask = self.warp_im(mask, M, self.img1.shape)

        combined_mask = numpy.max([self.get_face_mask(self.img1, self.landmarks1), warped_mask],axis=0)

        warped_im2 = self.warp_im(self.img2, M, self.img1.shape)

        warped_corrected_im2 = self.correct_colours(self.img1, warped_im2, self.landmarks1)

        output_im = self.img1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        
        return output_im


