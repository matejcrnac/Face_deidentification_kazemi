import cv2
import numpy
import collections

from FacialLandmarkDetection import *
from Database_loader import *

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
def find_closest_Image_sorted_list(image_positions, templatePositions, k=1):
    minScore = 1111111111111111
    distances = {}
    #N = len(image_positions)
    i=0
    for template in templatePositions:
        
        #difference = calculate_face_difference(image_positions, template)
        #difference = calculate_face_difference_all_distances(image_positions, template)
        difference = calculate_face_difference_specific_distances(image_positions, template)
        distances[difference] = i
        if difference < minScore:
            minScore = difference
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
    
    templates_database_orig = "/home/matej/Diplomski/baze/Templates/baza_templates"
    templates_similarity_test = "/home/matej/Diplomski/baze/Templates/templates_similarity_test"
    templates_database = templates_similarity_test
    
    
    imagePath_same1 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/000/000_1_1.ppm" # sve dobro,
    imagePath_same2 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/001/001_1_1.ppm" # sve dobro
    imagePath_same3 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/002/002_1_1.ppm" # spec bolji (k4 prob)
    imagePath_same4 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/004/004_1_1.ppm" # spec bolji od all dist
    imagePath_same5 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/005/005_1_1.ppm" # spec najbolji
    imagePath_same6 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/006/006_1_1.ppm" # k3 problem
    imagePath_same7 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/007/007_1_1.ppm" # nije dobar
    imagePath_same8 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/008/008_1_1.ppm" # sve dobro
    imagePath_same9 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/009/009_1_1.ppm" # sve dobro
    imagePath_same10 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/010/010_1_1.ppm" # sve dobro
    
    image_path_man_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_no_glasses/143_1_1.ppm"
    image_path_man_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_glasses/113_1_1.ppm"
    image_path_woman_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_no_glasses/154_1_1.ppm"
    image_path_woman_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_glasses/250_1_1.ppm"
    
    imagePath = imagePath_same10  #chose image to use!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
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
    k_list = [1, 2, 3,4, 5]
    for k in k_list:
        index = sorted_closest_indexes[k-1][0]
        distance_val = sorted_closest_indexes[k-1][1]
        template_path = template_paths[index]
        detector = FacialLandmarkDetector(template_path)
        img = detector.detectFacialLandmarks_get_image()
        
        showImage_more(img=img, text=str(k) + "-" + str(distance_val), gray=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
