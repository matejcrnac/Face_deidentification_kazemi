import dlib
import cv2
import numpy
import collections

from FacialLandmarkDetection import *
from Database_loader import *


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
#Method load image from database, and calculates facial landmarks for given image, and shows image if parameter is set to True
def loadDatabaseImage_CalculateFacialLandmarks(database_folder, imagePath, showImages=False):
    #loader = DatabaseLoaderXMVTS2(database_folder)
    detector = FacialLandmarkDetector(imagePath)
    positions = detector.detectFacialLandmarks(True)
    if showImages==True:
        detector.showImage()
    return positions

#Method is used to find closes template image from given image based on positions
#k - parameter which determines whichi image will be retured. if k=1, closes, if k=4, 4th closest
def find_closest_Image_sorted_list(image_positions, templatePositions, k=1):
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
    distances = collections.OrderedDict(sorted(distances.items()))
    index_sorted_list = []
    for key, value in distances.items():
        index_sorted_list.append((value, key))

    return index_sorted_list
#shows image inside a windows
def showImage_more(img,text,  gray=False):
    if gray==True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.putText(img,text , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    window_name = "window_" + text
    cv2.imshow(window_name,img)
    #cv2.waitKey(0)


if __name__ == "__main__":
    
    templates_database = "/home/matej/Diplomski/baze/Templates/baza_templates"
    imagePath_same1 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/000/000_1_1.ppm"
    imagePath_same2 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/003/003_1_1.ppm"
    imagePath_same3 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/004/004_1_1.ppm"
    imagePath_same4 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/041/041_1_1.ppm"
    imagePath_same5 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/134/134_1_1.ppm"
    
    image_path_man_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_no_glasses/143_1_1.ppm"
    image_path_man_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_glasses/113_1_1.ppm"
    image_path_woman_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_no_glasses/154_1_1.ppm"
    image_path_woman_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_glasses/250_1_1.ppm"
    
    imagePath = image_path_man_no_glasses  #chose image to use!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    destination_deident = ""
    
    (base, templates_positions) = loadTemplatesPositions(templates_database)#load positions of template from txt file
    
    base_position = numpy.matrix([[p[0], p[1]] for p in base])
    
    detector = FacialLandmarkDetector(imagePath)
    detector.warpe_image(base_positions = base_position)
    
    image_positions = detector.detectFacialLandmarks(draw=False, normalize = True, numpy_format = False)
    
    sorted_closest_indexes = find_closest_Image_sorted_list(image_positions, templates_positions, k=4) #find k-th closest image index
    
    print("Sorted template indexes based on distance from image")
    print(sorted_closest_indexes)
    
    #show original image
    detector = FacialLandmarkDetector(imagePath)
    image_original = detector.detectFacialLandmarks_get_image()

    
    showImage_more(img=image_original, text="original", gray=False)
    #show template images
    
    template_paths = getTemplatePaths(templates_database, extension="ppm")
    
    print(template_paths[sorted_closest_indexes[0][0]])
    k_list = [1, 2, 3,10, 15]
    for k in k_list:
        index = sorted_closest_indexes[k-1][0]
        distance_val = sorted_closest_indexes[k-1][1]
        template_path = template_paths[index]
        detector = FacialLandmarkDetector(template_path)
        img = detector.detectFacialLandmarks_get_image()
        
        showImage_more(img=img, text=str(k) + "-" + str(distance_val), gray=False)
    cv2.waitKey(0)
