from FacialLandmarkDetection import *
from Database_loader import *
import collections

#method shows all database images in windows

def showAllDatabaseImages(database_folder, extension="ppm", imageNum=""):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase(extension, imageNum)
    print(images_paths)
    for imagePath in images_paths:
        detector = FacialLandmarkDetector(imagePath)
        detector.showImage()
def saveImage(imagePath, destination):
    imgName = imagePath.split("/")[-1]
    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    cv2.imwrite(destination+"/"+imgName,img)

def storeDatabaseImagesToDestination(database_folder, destination, extension="ppm", ImageNum=""):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase(extension, ImageNum)
    for imagePath in images_paths:
        saveImage(imagePath, destination)
    print("Saving images finished!")
def findFacialLandmarksOnTemplateImages(templates_folder, destination, showImages = False, store=False, storePositions = False):
    loader = DatabaseLoaderXMVTS2(templates_folder)
    images_paths = loader.loadDatabase("ppm", "")
    landmarksPositions = []
    for imagePath in images_paths:
        detector = FacialLandmarkDetector(imagePath)
        positions = detector.detectFacialLandmarks(True)
        landmarksPositions.append(positions)
        if store == True:
            detector.saveImage(destination)
        if storePositions == True:
            text = ' '.join('%s %s' % x for x in positions)
            imageName = imagePath.split("/")[-1].split(".")[0]
            dirName = "/".join(imagePath.split("/")[:-1])
            f=open(dirName + "/" + imageName+".txt",'w')
            f.write(text)
            print("Stored result for image " + imageName)
            f.close()
        if showImages==True:
            detector.showImage()
    print("Finished finding facial landmarks on templates.")
def findFacialLandmarksOnTemplateImages_EyeRegion(templates_folder, destination, showImages = False, store=False, storePositions = False):
    loader = DatabaseLoaderXMVTS2(templates_folder)
    images_paths = loader.loadDatabase("ppm", "")
    landmarksPositions = []
    for imagePath in images_paths:
        detector = FacialLandmarkDetector(imagePath)
        ROI = detector.extractFacePart("EyeRegion")
        #landmarksPositions.append(positions)
        #if store == True:
         #   detector.saveImage(destination)
        #if storePositions == True:
         #   text = ' '.join('%s %s' % x for x in positions)
          #  imageName = imagePath.split("/")[-1].split(".")[0]
          #  dirName = "/".join(imagePath.split("/")[:-1])
            #f=open(dirName + "/" + imageName+".txt",'w')
            #f.write(text)
            #print("Stored result for image " + imageName)
            #f.close()
        if showImages==True:
            cv2.imshow('image',ROI)
            cv2.waitKey(0)
    print("Finished finding facial landmarks on templates.")
def getTemplatePaths(templates_folder, extension):
    fileNames = []
    for root, dirs, files in os.walk(templates_folder):
        for file in files:
            if file.endswith(extension):
                fName = os.path.join(root, file)
                fileNames.append(fName)
    fileNames = sorted(fileNames)
    return fileNames
def loadTemplatesPositions(templates_folder):
    positions = []
    fileNames = getTemplatePaths(templates_folder, "txt")
    for fileName in fileNames:
        position = []
        f = open(fileName,"r")
        line = f.readline()
        tempPos = line.split(" ")
        for i in range(0, len(tempPos), 2):
            position.append((float(tempPos[i]), float(tempPos[i+1])))
        positions.append(position)
        f.close()

    return positions
def find_closest_Image(image_positions, templatePositions, k=1):
    closest = -1
    minScore = 1111111111111111
    distances = {}
    N = len(image_positions)
    i=0
    for template in templatePositions:
        difference = sum(sum(numpy.subtract(image_positions, template)**2)/N)
        distances[difference] = i
        if difference < minScore:
            minScore = difference
            closest = i
        i += 1
    print(distances)
    distances = collections.OrderedDict(sorted(distances.items()))
    i=0
    for key, value in distances.items():
        if i==(k-1):
            return value
        i +=1

    return closest
def loadDatabaseImage_CalculateFacialLandmarks(database_folder, imagePath, showImages=False):
    loader = DatabaseLoaderXMVTS2(database_folder)
    detector = FacialLandmarkDetector(imagePath)
    positions = detector.detectFacialLandmarks(True)
    if showImages==True:
        detector.showImage()
    return positions
def getImagePath(database_folder, imageName):
    loader = DatabaseLoaderXMVTS2(database_folder)
    imagePath = loader.imagePathFinder(imageName, "")
    return imagePath
def replaceEyeRegionOfImageWithTemplate(imagePath, templatePath, showImage=True):
    detectorImage = FacialLandmarkDetector(imagePath)
    detectorTemplate = FacialLandmarkDetector(templatePath)
    eyeRegionImage = detectorImage.extractFacePart("EyeRegion")
    eyeRegionTemplate = detectorTemplate.extractFacePart("EyeRegion")
    resized_image = cv2.resize(eyeRegionTemplate, (eyeRegionImage.shape[1], eyeRegionImage.shape[0]) )
    if showImage == True:
        detectorImage.showImage()
    detectorImage.replaceImagePart(resized_image)
    if showImage == True:
        detectorImage.showImage(gray=False)
    return detectorImage.getImage()
def databaseToGrayScale(database_folder, destinationFolder, numImages, firstImageNum = 0,  lastImageNum = 1, grayScale=True):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase("ppm", "")
    dir_i = 1
    i = firstImageNum
    print("Building gray database ....")
    for imagePath in images_paths:
        if dir_i > numImages:
            break
        if grayScale == True:
            img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(imagePath,cv2.IMREAD_COLOR)
        #resized_image = cv2.resize(img, (92, 112))
        imagePathArray = imagePath.split("/")
        imageName = imagePathArray[-1].split("_")[1] + ".ppm"
        dir = str(dir_i)
        i += 1
        if i >=lastImageNum:
            dir_i += 1
            i = 0
        destination = destinationFolder + "_" + str(numImages) + "/s" + dir
        if not os.path.exists(destination):
            os.makedirs(destination)
        destination = destination + "/" + imageName
        cv2.imwrite(destination,img)
        #print("Image " + imagePath + "writen to " + destination)
    print("Finished building gray database")

def deidentifyImages(database_folder, destinationFolder, templates_database, numImages, gray=False):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase("ppm", "1")
    dir_i = 1
    i = 0
    print("Building gray database ....")
    for imagePath in images_paths:
        if dir_i > numImages:
            break
        #imagePathArray = imagePath.split("/")
        imageName = "4" + ".ppm"
        dir = str(dir_i)
        i += 1
        if i >=1:
            dir_i += 1
            i = 0
        destination = destinationFolder + "_" + str(numImages) + "/s" + dir
        if not os.path.exists(destination):
            os.makedirs(destination)
        destination = destination + "/" + imageName

        templates_positions = loadTemplatesPositions(templates_database)
        image_positions = loadDatabaseImage_CalculateFacialLandmarks(destination, imagePath, False)
        closest_Index = find_closest_Image(image_positions, templates_positions, 1)
        closest_Image_path = getTemplatePaths(templates_database, "ppm")[closest_Index]
        print(closest_Image_path)
        replacedImg = replaceEyeRegionOfImageWithTemplate(imagePath, closest_Image_path, showImage=False)

        if gray == True:
            replacedImg = cv2.cvtColor(replacedImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(destination,replacedImg)
        print("Image " + imagePath + "writen to " + destination)
    print("Finished building gray database")
if __name__ == "__main__":

    #OPERATIONS
    # 1 - find closes image from image, do deidnetification
    # 2 - database to grayFormat
    # 3 - deidentification on more images
    # 4 - store database image to destination
    operation = 1


    XMVTS2_database = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/baza_XMVTS2"
    destination = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/baza_deidentification_Images"
    templates_database = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/baza_templates"
    templates_destination = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/KazemiTemplates"
    XMVTS2_gray_faceReq = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/baza_XMVTS2_gray_facereq"
    deidentifiedImages = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/deidentifiedImages"

    databaseImages_2 = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/baza_deidentification_Images_2"
    databaseImages_3 = "/home/matej/MEGAsync/Diplomski_projekt/Project_Deidentification_Kazemi/baze/baza_deidentification_Images_3"
    if operation == 1:

        #findFacialLandmarksOnTemplateImages(templates_database, templates_destination, False, False, True)
        #findFacialLandmarksOnTemplateImages_EyeRegion(templates_database, templates_destination, True, False, False)
        imagePath = getImagePath(destination,"008")
        templates_positions = loadTemplatesPositions(templates_database)
        image_positions = loadDatabaseImage_CalculateFacialLandmarks(destination, imagePath, False)
        closest_Index = find_closest_Image(image_positions, templates_positions, 1)
        closest_Image_path = getTemplatePaths(templates_database, "ppm")[closest_Index]
        print(closest_Image_path)
        replaceEyeRegionOfImageWithTemplate(imagePath, closest_Image_path)
    elif operation == 2:
        databaseToGrayScale(databaseImages_3, deidentifiedImages, 40,firstImageNum = 2,  lastImageNum = 3, grayScale=True)
    elif operation == 3:
        deidentifyImages(destination, deidentifiedImages, templates_database, 40, gray=True)
    elif operation == 4:
        storeDatabaseImagesToDestination(XMVTS2_database, databaseImages_3, extension="ppm", ImageNum="3")
