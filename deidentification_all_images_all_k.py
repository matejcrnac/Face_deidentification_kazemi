from FacialLandmarkDetection import *
from Database_loader import *
import collections
from FaceSwaper import *
import numpy as np
from voting_gender_classifier import *

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

database_deidentification_folder = "/home/matej/Diplomski/baze/deidentification_database/baza_deidentification_Images"


templates_database = "/home/matej/Diplomski/baze/Templates/baza_templates"

templates_man_no_glasses = "/home/matej/Diplomski/baze/Templates/man_no_glasses"
templates_man_no_glasses_destination = "/home/matej/Diplomski/baze/Templates/man_no_glasses_Kazemi"


templates_destination = "/home/matej/Diplomski/baze/KazemiTemplates"

templates_database_man = "/home/matej/Diplomski/baze/Templates/man_all"
templates_database_woman = "/home/matej/Diplomski/baze/Templates/woman_all"

XMVTS2_gray_faceReq = "/home/matej/Diplomski/baze/baza_XMVTS2_gray_facereq"

deidentifiedImages = "/home/matej/Diplomski/baze/deidentification_destination"
databaseImages_2 = "/home/matej/Diplomski/baze/baza_deidentification_Images_2"
databaseImages_3 = "/home/matej/Diplomski/baze/baza_deidentification_Images_3"

EYES_EDGES = [[36, 39], [42, 45], [36, 42], [36, 45], [39, 42], [39, 45]]
EYES_IRIS = [[37, 41], [37, 40], [38, 40], [38, 41], [43, 47], [43, 46], [44, 46], [44, 47]]
EYEBROWS = [[17, 18], [18, 19], [19, 20], [20, 21], [17, 21], [22, 23], [23, 24], [24, 25], [25, 26], [22, 26], [17, 26]]
EYES_EYEBROWS = [[17, 36], [21, 39], [22, 42], [26, 45], [17, 39], [21, 36], [22, 45], [26, 42]]
NOSE = [[27, 30], [30, 33], [31, 32], [32, 34], [34, 35]]
EYES_NOSE = [[33, 36], [33, 39], [33, 42], [33, 45]]
MOUTH_OUTER = [[48, 54], [49, 59], [50, 58], [51, 57], [52, 56], [53, 55], [49, 55], [53, 59]]
MOUTH_THICKNESS = [[51, 61], [57, 64], [49, 60], [50, 60], [52,62], [53, 62], [55, 63], [56, 63], [58, 65], [59, 65]]
NOSE_MOUTH = [[33, 48], [33, 54], [33, 51], [31, 48], [35, 54]]
EYES_MOUTH = [[36, 48], [39, 47], [42, 54], [45, 54]]

SPECIFIC_DISTANCES = EYES_EDGES + EYES_IRIS + EYEBROWS + EYES_EYEBROWS + NOSE + EYES_NOSE + MOUTH_OUTER + MOUTH_THICKNESS + NOSE_MOUTH + EYES_MOUTH


if __name__ == "__main__":

    database_folder = database_deidentification_folder
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase("ppm", "1")
    
    #method = "landmarks_diff"
    #method = "landmarks_distances_all"
    method = "landmar_distances_spec"

    destination = destination + "/deidentification_destination_all_images_all_k_" + method + "/"
    if not os.path.exists(destination):
        os.makedirs(destination)

    model1_path = "gender_models/gender_detection_keras.model"
    model2_path = "gender_models/svm_model.pkl"
    model3_path = "gender_models/generator_model.hdf5"

    gender_detector = VotingGenderDetector(model1_path, model2_path, model3_path)

    for imagePath in images_paths:
        imageName = imagePath.split("/")[-1].split("_")[0]
        destination_new = destination + imageName + "/"
        if not os.path.exists(destination_new):
            os.makedirs(destination_new)

        #find gender
        gender = gender_detector.predict_label(imagePath)
        if gender == "man":
            templates_database = templates_database_man
        else:
            templates_database = templates_database_woman       
        (base, templates_positions) = loadTemplatesPositions(templates_database)#load positions of template from txt file
        if len(templates_positions) == 0:
            print("No loaded templates!! Please check if you have generated face landmarks for templates.")
            exit()
        base_position = numpy.matrix([[p[0], p[1]] for p in base])
        
        detector = FacialLandmarkDetector(imagePath)
        detector.warpe_image(base_positions = base_position)
        
        image_positions_non_norm = detector.detectFacialLandmarks(draw=False, normalize = True, numpy_format = False)
        image_positions = detector.normalize(image_positions_non_norm)
        
        sorted_closest_indexes = find_closest_Image_sorted_list(image_positions, templates_positions,method = method,  k=4) #find k-th closest image index

        imageName = imagePath.split("/")[-1].split("_")[0]
        
        
        templates_number = len(sorted_closest_indexes)
        
        for k in range(1, templates_number+1):
            
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
            
            imname = destination_new + imageName + "k_" + str(k) + ".ppm"
            cv2.imwrite(imname, out)
            print("Finished image" + str(k))
            #cv2.imshow("deidentified", image/255.)
                        
            
            #cv2.waitKey(0)
        del image
        del img_orig
        del img_orig2
        del out
        del detector



