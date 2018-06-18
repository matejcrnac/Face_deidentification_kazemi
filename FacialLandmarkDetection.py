
import dlib
import cv2
import numpy

class FacialLandmarkDetector:

    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        self.dets = None
        self.shape = None
    #Detects frontal face on image
    #Returns dlib.rectangle object
    #agrument draw decides if rectangle is drawn on image
    def detect_frontal_face(self, draw=False):
        detector = dlib.get_frontal_face_detector()
        self.dets = detector(self.img, 1)
        for k, d in enumerate(self.dets):
            if draw==True:
                cv2.rectangle(self.img,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
        return d
    #shows image inside a windows
    def showImage(self, gray=False):
        imgShow = self.img
        if gray==True:
            imgShow = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.putText(imgShow,self.image_path , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('image',imgShow)
        cv2.waitKey(0)
    def getImage(self):
        return self.img
    def saveImage(self, destination, additional_name = ""):
        imgName = self.image_path.split("/")[-1] + additional_name
        cv2.imwrite(destination+"/"+imgName,self.img)
    def normalize(self, parts):
        mean = (sum([value[0] for value in parts]) / float(len(parts)), sum([value[1] for value in parts]) / float(len(parts)))
        
        #subtract every point with mean
        newPoint = []
        for part in parts:
            #think about normalizes by variance
            newPoint.append(tuple(numpy.subtract(part, mean)))
        #find max point value after subtraction
        normalized = []
        max_X = abs(max(newPoint, key = lambda item:abs(item[0]))[0])
        max_Y = abs(max(newPoint, key = lambda item:abs(item[1]))[1])
        maxValue = max(max_X, max_Y)
        
        for point in newPoint:
            #think about normalizes by variance
            normalized.append(tuple(numpy.divide(point, maxValue)))
        return normalized
        
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
                             
    #Method uses affine transformation on image
    def warp_im(self, im, M, dshape):
        output_im = numpy.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im
        
    #Method is used to warp current image based on given positions
    def warpe_image(self, base_positions):
        image_landmark_positions = self.detectFacialLandmarks(draw=False, normalize=False, numpy_format = True)
        #image_landmark_positions = numpy.matrix([[p[0], p[1]] for p in image_landmark_positions])
        
        M = self.transformation_from_points(base_positions,image_landmark_positions)
        self.img = self.warp_im(self.img, M, self.img.shape)
        

    #detects facial landmarks based
    #returns list of tuples of (x,y) which represent 68 landmark points
    def detectFacialLandmarks(self, draw, normalize=True, numpy_format = False):
        #shape_predictor_68_face_landmarks.dat can be downloaded from
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.parts = []
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
        d = self.detect_frontal_face()
        self.shape = predictor(self.img, d)

        for i in range(self.shape.num_parts):
            self.parts.append((self.shape.part(i).x,self.shape.part(i).y))
            if draw==True:
                cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if normalize==True:
            self.parts = self.normalize(self.parts)
        if numpy_format == False:
            return self.parts
        return numpy.matrix([[p[0], p[1]] for p in self.parts])
        
        #detects facial landmarks based
    #returns list of tuples of (x,y) which represent 68 landmark points
    def detectFacialLandmarks_get_image(self):
        #shape_predictor_68_face_landmarks.dat can be downloaded from
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.parts = []
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
        d = self.detect_frontal_face()
        self.shape = predictor(self.img, d)

        for i in range(self.shape.num_parts):
            self.parts.append((self.shape.part(i).x,self.shape.part(i).y))
            cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        return self.img

    def getFacialLandmarksOfFacePart(self, faceParts, draw=False):
        if self.shape == None:
            self.parts = self.detectFacialLandmarks(False, normalize=False)
        foundParts = []
        if "Mouth" in faceParts:
            for i in range(48, 68):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "RightEyebrow" in faceParts:
            for i in range(17, 22):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "LeftEyebrow" in faceParts:
            for i in range(22, 27):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "RightEye" in faceParts:
            for i in range(36, 42):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "LeftEye" in faceParts:
            for i in range(42, 48):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "Nose" in faceParts:
            for i in range(27, 36):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "Jaw" in faceParts:
            for i in range(0, 17):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        return foundParts
    def extractFacePart(self, facePart):
        self.region = None
        if facePart == "EyeRegion":
            parts = self.getFacialLandmarksOfFacePart(["RightEye", "LeftEye", "RightEyebrow", "LeftEyebrow"])
        if facePart == "Nose":
            parts = self.getFacialLandmarksOfFacePart(["Nose"])
        if facePart == "Mouth":
            parts = self.getFacialLandmarksOfFacePart(["Mouth"])
        if facePart == "Face":
            parts = self.getFacialLandmarksOfFacePart(["RightEye", "LeftEye", "RightEyebrow", "LeftEyebrow", "Mouth"])
            
        self.top, self.left, self.bottom, self.right = maxRectangle(parts)
        self.bottom += 10
        self.region = self.img[self.top:self.bottom, self.left:self.right]

        return self.region
    def replaceImagePart(self, newROI):


        #calculate skin color of original image

#        top = self.region.shape[0] - 10
#        botom = self.region.shape[0]
#        right = self.region.shape[1]/2 + 10
#        left = self.region.shape[1]/2 - 10
#
#        bottomRegionPart = self.region[top:botom, left:right]
#        sum0 = 0
#        sum1 = 0
#        sum2 = 0
#        for i in range(bottomRegionPart.shape[0]):
#            for j in range(bottomRegionPart.shape[1]):
#                sum0 += bottomRegionPart[i][j][0]
#                sum1 += bottomRegionPart[i][j][0]
#                sum2 += bottomRegionPart[i][j][0]
#        avg0 = sum0/(bottomRegionPart.shape[0]*bottomRegionPart.shape[1])
#        avg1 = sum1/(bottomRegionPart.shape[0]*bottomRegionPart.shape[1])
#        avg2 = sum2/(bottomRegionPart.shape[0]*bottomRegionPart.shape[1])
#
#        #change skin color of new part
#
#        for i in range(newROI.shape[0]):
#            for j in range(newROI.shape[1]):



        self.img[self.top:self.bottom, self.left:self.right] = newROI
        #self.showImage()

        #bottom
        newTop = self.bottom - 5
        newBottom = self.bottom + 5
        regionBottom = self.img[newTop:newBottom, self.left:self.right]
        skinColor = int(sum(numpy.sum(regionBottom[-1, :],axis=0) / len(regionBottom[-1, :]))/3)
        blur = cv2.bilateralFilter(regionBottom,15,skinColor-5,skinColor+5)
        self.img[newTop:newBottom, self.left:self.right] = blur

        #top
        newTop = self.top - 5
        newBottom = self.top + 5
        regionTop = self.img[newTop:newBottom, self.left:self.right]
        skinColor = int(sum(numpy.sum(regionTop[-1, :],axis=0) / len(regionTop[-1, :]))/3)
        blur = cv2.bilateralFilter(regionTop, 15,skinColor-5,skinColor+5)
        self.img[newTop:newBottom, self.left:self.right] = blur

        #left
        newLeft = self.left - 5
        newRight = self.left + 5
        regionTop = self.img[self.top:self.bottom, newLeft:newRight]
        skinColor = int(sum(numpy.sum(regionTop[-1, :],axis=0) / len(regionTop[-1, :]))/3)
        blur = cv2.bilateralFilter(regionTop,15,skinColor-5,skinColor+5)
        self.img[self.top:self.bottom, newLeft:newRight] = blur

        #right
        newLeft = self.right - 5
        newRight = self.right + 5
        regionTop = self.img[self.top:self.bottom, newLeft:newRight]
        skinColor = int(sum(numpy.sum(regionTop[-1, :],axis=0) / len(regionTop[-1, :]))/3)
        blur = cv2.bilateralFilter(regionTop,15,skinColor-5,skinColor+5)
        self.img[self.top:self.bottom, newLeft:newRight] = blur


def maxRectangle(parts):
    maxTop = 1111111111
    maxLeft = 111111111
    maxBottom = -100
    maxRight = -100
    for part in parts:
        if part[0] < maxLeft:
            maxLeft = part[0]
        if part[0] > maxRight:
            maxRight = part[0]
        if part[1] < maxTop:
            maxTop = part[1]
        if part[1] > maxBottom:
            maxBottom = part[1]
    return maxTop, maxLeft, maxBottom, maxRight

if __name__ == "__main__":
    image_name = "/home/matej/Diplomski/baze/baza_XMVTS2/000/000_1_1.ppm"
    detector = FacialLandmarkDetector(image_name)
    #detector.detect_frontal_face(True)
    detector.showImage()
    parts = detector.detectFacialLandmarks(True)
    detector.showImage()
    #print(parts)
    #foundParts = detector.getFacialLandmarksOfFacePart(["Nose", "Mouth"], True)
    #detector.showImage()
    #print(foundParts)
    #ROI = detector.extractFacePart("EyeRegion")
    #cv2.imshow('image',ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
