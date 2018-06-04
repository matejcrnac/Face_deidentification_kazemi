from FacialLandmarkDetection import *
from Database_loader import *
import collections
from FaceSwaper import *
import numpy as np

#This program has 5 operations. Chose operation int the bottom. Description of each operation is given in the bottom.



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
    base = []
    for fileName in fileNames:
        position = []
        f = open(fileName,"r")
        line = f.readline()
        tempPos = line.split(" ")
        if "base" in fileName:
            print("Found base")
            for i in range(0, len(tempPos[:-1]), 2):
                base.append((int(tempPos[i]), int(tempPos[i+1])))
            continue
            
        for i in range(0, len(tempPos), 2):
            position.append((float(tempPos[i]), float(tempPos[i+1])))
        positions.append(position)
        f.close()

    return (base, positions)
    
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

def calculate_face_difference(image1, image2):
    image1 = numpy.matrix([[p[0], p[1]] for p in image1])
    image2 = numpy.matrix([[p[0], p[1]] for p in image2])
    
    N = len(image1)
    return numpy.sum(numpy.power(numpy.subtract(image1, image2),2))/N
    
def calculate_face_difference_all_distances(image1, image2):
    image1 = numpy.matrix([[p[0], p[1]] for p in image1])
    image2 = numpy.matrix([[p[0], p[1]] for p in image2])
    
    
    N = len(image1)
    iter = 0
    sum = 0
    for i in range(N-1):
        for j in range(i+1, N):
            iter += 1
            sum += numpy.sum(numpy.power((image1[i] - image1[j]) - (image2[i] - image2[j]), 2))
            
    return sum/iter
    
def calculate_face_difference_specific_distances(image1, image2):
    image1 = numpy.matrix([[p[0], p[1]] for p in image1])
    image2 = numpy.matrix([[p[0], p[1]] for p in image2])
    
    iter = 0
    sum = 0
    for pair in SPECIFIC_DISTANCES:
        iter += 1
        sum += numpy.sum(numpy.power((image1[pair[0]] - image1[pair[1]]) - (image2[pair[0]] - image2[pair[1]]), 2))
            
    return sum/iter

#Method is used to find closes template image from given image based on positions
#k - parameter which determines whichi image will be retured. if k=1, closes, if k=4, 4th closest
def find_closest_Image_sorted_list(image_positions, templatePositions, method = "landmarks_diff", k=1):
    minScore = 1111111111111111
    distances = {}
    #N = len(image_positions)
    i=0
    for template in templatePositions:
        if method == "landmarks_diff":
            difference = calculate_face_difference(image_positions, template)
        elif method == "landmarks_distances_all":
            difference = calculate_face_difference_all_distances(image_positions, template)
        elif method == "landmar_distances_spec":
            difference = calculate_face_difference_specific_distances(image_positions, template)
        else:
            printf("Wrong distance method!!!")
            exit()
        distances[difference] = i
        if difference < minScore:
            minScore = difference
        i += 1
    distances = collections.OrderedDict(sorted(distances.items()))
    index_sorted_list = []
    for key, value in distances.items():
        index_sorted_list.append((value, key))

    return index_sorted_list

        
#Method returns image path for given name of image (000,001 ...)
def getImagePath(database_folder, imageName):
    loader = DatabaseLoaderXMVTS2(database_folder)
    imagePath = loader.imagePathFinder(imageName, "")
    return imagePath


#shows image inside a windows
def showImage_more(img,text,  gray=False):
    if gray==True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.putText(img,text , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    window_name = "window_" + text
    cv2.imshow(window_name,img)
    #cv2.waitKey(0)


#Used paths
man_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_no_glasses"
man_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_glasses"
woman_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_no_glasses"
woman_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_glasses"


XMVTS2_database = "/home/matej/Diplomski/baze/baza_XMVTS2"
destination = "/home/matej/Diplomski/baze/deidentification_destination"
templates_database = "/home/matej/Diplomski/baze/Templates/baza_templates"


templates_destination = "/home/matej/Diplomski/baze/KazemiTemplates"

XMVTS2_gray_faceReq = "/home/matej/Diplomski/baze/baza_XMVTS2_gray_facereq"

deidentifiedImages = "/home/matej/Diplomski/baze/deidentification_destination"
databaseImages_2 = "/home/matej/Diplomski/baze/baza_deidentification_Images_2"
databaseImages_3 = "/home/matej/Diplomski/baze/baza_deidentification_Images_3"

SPECIFIC_DISTANCES = [[36, 39], [42, 45], [17, 36], [26, 45], [21, 39], [22, 42], [17, 21], [22, 26], [33, 39], [33, 42], 
                        [39, 42], [36, 45], [49, 55], [48, 54], [0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], 
                        [7, 9], [57, 8], [50, 61], [51, 62], [52, 63], [56, 65], [57, 66], [58, 67], [32, 50], [34, 52], 
                        [3, 46], [54, 13]]


if __name__ == "__main__":

    imagePath = getImagePath(man_no_glasses,"001") #image folder,image name
    
    (base, templates_positions) = loadTemplatesPositions(templates_database)#load positions of template from txt file
    if len(templates_positions) == 0:
        print("No loaded templates!!")
        exit()
    base_position = numpy.matrix([[p[0], p[1]] for p in base])
    
    detector = FacialLandmarkDetector(imagePath)
    detector.warpe_image(base_positions = base_position)
    
    image_positions_non_norm = detector.detectFacialLandmarks(draw=False, normalize = True, numpy_format = False)
    image_positions = detector.normalize(image_positions_non_norm)
    
    method = "landmarks_diff"
    #method = "landmarks_distances_all"
    #method = "landmar_distances_spec"
    sorted_closest_indexes = find_closest_Image_sorted_list(image_positions, templates_positions,method = method,  k=4) #find k-th closest image index

    imageName = imagePath.split("/")[-1].split("_")[0]
    
    destination = destination + "/deidentification_destination" + method  +"_"+ imageName + "/"
    if not os.path.exists(destination):
        os.makedirs(destination)
        
    for k in range(1, 21):
        
        closest_Index = sorted_closest_indexes[k-1][0]
        
        closest_Image_path = getTemplatePaths(templates_database, "ppm")[closest_Index]
        
        faceswaper = FaceSwap(imagePath, closest_Image_path)
        image = faceswaper.swap_face()
        
         
        img_orig = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        cv2.putText(img_orig,"original" , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        img_orig2 = cv2.imread(closest_Image_path, cv2.IMREAD_COLOR)
        cv2.putText(img_orig2,"original2" , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image,"k="+str(k) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        out = np.concatenate((img_orig, image), axis=1)

        out = np.concatenate((out, img_orig2), axis=1)
        
        imname = destination + imageName + "k_" + str(k+1) + ".ppm"
        cv2.imwrite(imname, out)
        print("Finished image" + str(k))
        #cv2.imshow("deidentified", image/255.)
                    
        
        #cv2.waitKey(0)


