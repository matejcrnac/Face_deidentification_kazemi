#input one image
#detect facial landmarks
#show image with detected facial landmarks
#show image with normalized facial landmarks


import cv2
import numpy as np

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
    
def unnormalized_facial_landmarks_detect(imagePath):
    detector = FacialLandmarkDetector(imagePath)
    image_original = detector.getImage()
    shape = image_original.shape
    #get normalized landmarks on black background
    image_landmarks_norm = detector.detectFacialLandmarks(draw=False, normalize=True)
    
    image_orig_black_white_norm = np.zeros((int(shape[0]/3), int(shape[1]/3)), dtype=np.float64)
    
    max_shape = np.max(shape)
    for position in image_landmarks_norm:
        x = ((position[0] * (max_shape/8))+max_shape/8).astype(np.int32)
        y = ((position[1] * (max_shape/8))+max_shape/8).astype(np.int32)
        
        cv2.circle(image_orig_black_white_norm,(x,y), 1, (1,1,1), -1)
    return image_orig_black_white_norm


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
    
    imagePath = image_path_man_glasses  #chose image to use!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
        

    template_paths = getTemplatePaths(templates_database, extension="ppm")
    
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for k in k_list:
        image_orig_black_white_norm = unnormalized_facial_landmarks_detect(imagePath = template_paths[k-1])
        showImage_more(img=image_orig_black_white_norm, text=str(k) + "- image", gray=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
