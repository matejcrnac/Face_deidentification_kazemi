from FacialLandmarkDetection import *
from Database_loader import *
import collections

#This program has 5 operations. Chose operation int the bottom. Description of each operation is given in the bottom.


#Used paths
man_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_no_glasses"
man_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_glasses"
woman_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_no_glasses"
woman_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_glasses"


XMVTS2_database = "/home/matej/Diplomski/baze/baza_XMVTS2"
destination = "/home/matej/Diplomski/baze/baza_deidentification_Images"
templates_database = "/home/matej/Diplomski/baze/Templates/baza_templates"


templates_destination = "/home/matej/Diplomski/baze/KazemiTemplates"

XMVTS2_gray_faceReq = "/home/matej/Diplomski/baze/baza_XMVTS2_gray_facereq"

deidentifiedImages = "/home/matej/Diplomski/baze/deidentification_destination"
databaseImages_2 = "/home/matej/Diplomski/baze/baza_deidentification_Images_2"
databaseImages_3 = "/home/matej/Diplomski/baze/baza_deidentification_Images_3"

#--------------------------------------METHODS------------------------------------------------------------------------

#method shows all database images in windows
#database_folder - parameter represents name of folder in which database is stored
#extension - parameter shows extension of images. Default extension is .ppm
#imageNum - parameter shows number of image for each person. Ecah person has 4 images. This number determines which image
#is chosen
def showAllDatabaseImages(database_folder, extension="ppm", imageNum=""):
    loader = DatabaseLoaderXMVTS2(database_folder) # load database loader object
    images_paths = loader.loadDatabase(extension, imageNum) #get paths of all iamges in database
    print(images_paths)
    for imagePath in images_paths:      #for each image path
        detector = FacialLandmarkDetector(imagePath) #create FacialLandmarkDetector object for image
        detector.showImage()    # show image
        
#Method is used for saving image in given destination
#imagePath - parameter represents image path
#destination - parameter represents name of folder in which to store given image
def saveImage(imagePath, destination):
    imgName = imagePath.split("/")[-1]  #get image name from image path
    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)   # load image
    cv2.imwrite(destination+"/"+imgName,img)    #store image in destination, 

#Method stores all database images to given destination
#database_folder - parameter represents name of folder in which database is located
#destination - name of destination folder in which to store all images
#extension - represents extension of images
#imageNum - represents number of image which will be used
def storeDatabaseImagesToDestination(database_folder, destination, extension="ppm", ImageNum=""):
    loader = DatabaseLoaderXMVTS2(database_folder)  # create DatabaseLoaderXMVTS2 object for database
    images_paths = loader.loadDatabase(extension, ImageNum) # get all images paths
    for imagePath in images_paths:  #for each image path
        saveImage(imagePath, destination)   #save image to given destination
    print("Saving images finished!")
    
#Method is used to find facial landmarks for TEMPLATE IMAGES and stores them to txt file
#templates_folder - name of folder in which templates are stored
#destination - name of folder in which to store images
#showImages - True/False, True - show images on screen, False- do not show images on screen
#store - True/False , True- store image with found landmarks
#storePositions - save found positions in txt file
def findFacialLandmarksOnTemplateImages(templates_folder, destination, showImages = False, store=False, storePositions = False):
    loader = DatabaseLoaderXMVTS2(templates_folder) #create DatabaseLoaderXMVTS2 object for given database
    images_paths = loader.loadDatabase("ppm", "")   # get all images paths
    landmarksPositions = [] 
    for imagePath in images_paths:  #for each image path
        detector = FacialLandmarkDetector(imagePath)    #create FacialLandmarkDetector object for image
        positions = detector.detectFacialLandmarks(True)    # get positions for image with normalization set to True
        landmarksPositions.append(positions)
        if store == True:   #is True, save image to destination folder
            detector.saveImage(destination)
        if storePositions == True:  #if True save landmark positions to txt file in destination
            text = ' '.join('%s %s' % x for x in positions) #set positions as text string
            imageName = imagePath.split("/")[-1].split(".")[0]  #get image name
            dirName = "/".join(imagePath.split("/")[:-1])   # ger directory name
            f=open(dirName + "/" + imageName+".txt",'w')    
            f.write(text)   #write text to file
            print("Stored result for image " + imageName)
            f.close()
        if showImages==True:    #if true show image
            detector.showImage()
    print("Finished finding facial landmarks on templates.")
    
#Method is used to extract eye region from images
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
    
#Method is used to get paths for template images
def getTemplatePaths(templates_folder, extension):
    fileNames = []
    for root, dirs, files in os.walk(templates_folder):
        for file in files:
            if file.endswith(extension):
                fName = os.path.join(root, file)
                fileNames.append(fName)
    fileNames = sorted(fileNames)
    return fileNames
    
#Method is used to load positions of template images which are stored in txt files.
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
    
#Method is used to find closes template image from given image based on positions
#k - parameter which determines whichi image will be retured. if k=1, closes, if k=4, 4th closest
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
    
#Method load image from database, and calculates facial landmarks for given image, and shows image if parameter is set to True
def loadDatabaseImage_CalculateFacialLandmarks(database_folder, imagePath, showImages=False):
    loader = DatabaseLoaderXMVTS2(database_folder)
    detector = FacialLandmarkDetector(imagePath)
    positions = detector.detectFacialLandmarks(True)
    if showImages==True:
        detector.showImage()
    return positions
    
#Method returns image path for given name of image (000,001 ...)
def getImagePath(database_folder, imageName):
    loader = DatabaseLoaderXMVTS2(database_folder)
    imagePath = loader.imagePathFinder(imageName, "")
    return imagePath

#Method is used to replace eye region of given image with template image
def replaceEyeRegionOfImageWithTemplate(imagePath, templatePath, parts=["Mouth", "EyeRegion", "Nose"], showImage=True):
    detectorImage = FacialLandmarkDetector(imagePath)
    detectorTemplate = FacialLandmarkDetector(templatePath)
    
    if showImage == True:
        detectorImage.showImage()
        
    #EYE REGION
    if "EyeRegion" in parts:
        eyeRegionImage = detectorImage.extractFacePart("EyeRegion")
        eyeRegionTemplate = detectorTemplate.extractFacePart("EyeRegion")
        resized_image = cv2.resize(eyeRegionTemplate, (eyeRegionImage.shape[1], eyeRegionImage.shape[0]) )
        
        detectorImage.replaceImagePart(resized_image)
        if showImage == True:
            detectorImage.showImage(gray=False)
        
    #NOSE REGION
    if "Nose" in parts:
        noseRegionImage = detectorImage.extractFacePart("Nose")
        noseRegionTemplate = detectorTemplate.extractFacePart("Nose")
        resized_image = cv2.resize(noseRegionTemplate, (noseRegionImage.shape[1], noseRegionImage.shape[0]) )
        
        detectorImage.replaceImagePart(resized_image)
        if showImage == True:
            detectorImage.showImage(gray=False)
        
    #MOUTH
    if "Mouth" in parts:
        mouthRegionImage = detectorImage.extractFacePart("Mouth")
        mouthRegionTemplate = detectorTemplate.extractFacePart("Mouth")
        resized_image = cv2.resize(mouthRegionTemplate, (mouthRegionImage.shape[1], mouthRegionImage.shape[0]) )
        
        detectorImage.replaceImagePart(resized_image)
        if showImage == True:
            detectorImage.showImage(gray=False)
    if "Face" in parts:
        faceRegionImage = detectorImage.extractFacePart("Face")
        faceRegionTemplate = detectorTemplate.extractFacePart("Face")
        resized_image = cv2.resize(faceRegionTemplate, (faceRegionImage.shape[1], faceRegionImage.shape[0]) )
        
        detectorImage.replaceImagePart(resized_image)
        if showImage == True:
            detectorImage.showImage(gray=False)
            
    return detectorImage.getImage()

#Method turns whole database to gray images
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

#Method is used to deidentifyImages.
#numImages - parameter determines how many images will be deidentified
#gray - parameter determines if gray images will be used
def deidentifyImages(database_folder, destinationFolder, templates_database, numImages,  description,  parts=["Mouth", "EyeRegion", "Nose"], gray=False, write_original=True):
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase("ppm", "1")
    dir_i = 1
    i = 0
    
    for imagePath in images_paths:  #for each image
        if dir_i > numImages:
            break
        imagePathArray = imagePath.split("/")
        imageName = imagePathArray[-1].split("_")[0] + "_deidentified.ppm"
        image_orig_name = imagePathArray[-1].split("_")[0] + ".ppm"
        
        
        i += 1
        if i >=1:
            i = 0
        destination = destinationFolder + "/deidentification_destination" + "_" + str(numImages) +  "_" + description + "/"
        if not os.path.exists(destination):
            os.makedirs(destination)
        destination_deident = destination + "/" + imageName
        destination_orig = destination + "/" + image_orig_name
        
        templates_positions = loadTemplatesPositions(templates_database)#load positions of template from txt file
        
        image_positions = loadDatabaseImage_CalculateFacialLandmarks(destination_deident, imagePath, False) #calculate landmarks
        closest_Index = find_closest_Image(image_positions, templates_positions, k=4) #find k-th closest image index
        closest_Image_path = getTemplatePaths(templates_database, "ppm")[closest_Index] # get closest image path
        print(closest_Image_path)
        replacedImg = replaceEyeRegionOfImageWithTemplate(imagePath, closest_Image_path, parts, showImage=False) #replace eyeregion
        
        if gray == True:
            replacedImg = cv2.cvtColor(replacedImg, cv2.COLOR_BGR2GRAY)
            
        if write_original == True:
            #load original image
            original_imagE = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            cv2.imwrite(destination_orig,original_imagE)
        cv2.imwrite(destination_deident,replacedImg)
        print("Image " + imagePath + "writen to " + destination)
    print("Finished building gray database")
if __name__ == "__main__":

    #OPERATIONS
    # 1 - find closes image from image, do deidnetification
    # 2 - database to grayFormat
    # 3 - deidentification on more images
    # 4 - store database image to destination
    # 5 - find facial landmarks on template images
    
    
    operation = 3


    if operation == 1:

        #findFacialLandmarksOnTemplateImages(templates_database, templates_destination, False, False, True)
        #findFacialLandmarksOnTemplateImages_EyeRegion(templates_database, templates_destination, True, False, False)
        imagePath = getImagePath(man_no_glasses,"008") #image folder,image name
        templates_positions = loadTemplatesPositions(templates_database)
        image_positions = loadDatabaseImage_CalculateFacialLandmarks(man_no_glasses, imagePath, False)
        closest_Index = find_closest_Image(image_positions, templates_positions, 1)
        closest_Image_path = getTemplatePaths(templates_database, "ppm")[closest_Index]
        print(closest_Image_path)
        replaceEyeRegionOfImageWithTemplate(imagePath, closest_Image_path, parts = ["EyeRegion", "Nose", "Mouth"])
    elif operation == 2:
        databaseToGrayScale(databaseImages_3, deidentifiedImages, 40,firstImageNum = 2,  lastImageNum = 3, grayScale=True)
    elif operation == 3:
        deidentifyImages(man_no_glasses, deidentifiedImages, templates_database, 40, description = "man_no_glasses_face", parts = ["Face"], gray=False, write_original = True)
    elif operation == 4:
        storeDatabaseImagesToDestination(XMVTS2_database, databaseImages_3, extension="ppm", ImageNum="3")
    elif operation == 5:
        findFacialLandmarksOnTemplateImages(templates_database, templates_destination, False, False, True)


