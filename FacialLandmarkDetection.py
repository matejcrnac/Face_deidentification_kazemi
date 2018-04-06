
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
    def saveImage(self, destination):
        imgName = self.image_path.split("/")[-1]
        cv2.imwrite(destination+"/"+imgName,self.img)
    def normalize(self, parts):
        mean = (sum([value[0] for value in parts]) / float(len(parts)), sum([value[1] for value in parts]) / float(len(parts)))
        normalized = []
        for part in parts:
            #think about normalizes by variance
            normalized.append(tuple(numpy.subtract(part, mean)))
        return normalized

    #detects facial landmarks based
    #returns list of tuples of (x,y) which represent 68 landmark points
    def detectFacialLandmarks(self, draw, normalize=True):
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
        return self.parts
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
        region = None
        if facePart == "EyeRegion":
            parts = self.getFacialLandmarksOfFacePart(["RightEye", "LeftEye", "RightEyebrow", "LeftEyebrow"])
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
    image_name = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_XMVTS2/000/000_1_1.ppm"
    detector = FacialLandmarkDetector(image_name)
    #detector.detect_frontal_face(True)
    detector.showImage()
    #parts = detector.detectFacialLandmarks(True)
    #detector.showImage()
    #print(parts)
    #foundParts = detector.getFacialLandmarksOfFacePart(["Nose", "Mouth"], True)
    #detector.showImage()
    #print(foundParts)
    ROI = detector.extractFacePart("EyeRegion")
    cv2.imshow('image',ROI)
    cv2.waitKey(0)
