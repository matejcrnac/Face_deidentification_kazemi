from FacialLandmarkDetection import *
from Database_loader import *

#This program has 5 operations. Chose operation int the bottom. Description of each operation is given in the bottom.


#Used paths
man_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_no_glasses"
man_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/man_glasses"
woman_no_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_no_glasses"
woman_glasses = "/home/matej/Diplomski/baze/deidentification_database/Deidentification_main/woman_glasses"


XMVTS2_database = "/home/matej/Diplomski/baze/baza_XMVTS2"
destination = "/home/matej/Diplomski/baze/baza_deidentification_Images"
templates_database = "/home/matej/Diplomski/baze/Templates/baza_templates"


templates_destination = "/home/matej/Diplomski/baze/Templates/KazemiTemplates"

XMVTS2_gray_faceReq = "/home/matej/Diplomski/baze/baza_XMVTS2_gray_facereq"

deidentifiedImages = "/home/matej/Diplomski/baze/deidentification_destination"
databaseImages_2 = "/home/matej/Diplomski/baze/baza_deidentification_Images_2"
databaseImages_3 = "/home/matej/Diplomski/baze/baza_deidentification_Images_3"

#--------------------------------------METHODS------------------------------------------------------------------------

    
#Method is used to find facial landmarks for TEMPLATE IMAGES and stores them to txt file
#templates_folder - name of folder in which templates are stored
#destination - name of folder in which to store images
#showImages - True/False, True - show images on screen, False- do not show images on screen
#store - True/False , True- store image with found landmarks
#storePositions - save found positions in txt file
def findFacialLandmarksOnTemplateImages(templates_folder, destination, warp = True, showImages = False, store=False, storePositions = False):
    loader = DatabaseLoaderXMVTS2(templates_folder) #create DatabaseLoaderXMVTS2 object for given database
    images_paths = loader.loadDatabase("ppm", "")   # get all images paths
    landmarksPositions = []
    base_image = images_paths[0]
    
    if warp == True:
        base_detector = FacialLandmarkDetector(base_image)
        base_positions = base_detector.detectFacialLandmarks(draw=False, normalize=False, numpy_format = True)
        imageName = base_image.split("/")[-1].split(".")[0]  #get image name
        dirName = "/".join(base_image.split("/")[:-1])   # ger directory name
        numpy.savetxt(dirName + "/" + imageName+ "_base" + ".txt", base_positions,fmt='%d',  delimiter=' ', newline=' ', header='', footer='', comments='# ', encoding=None)
        print("Stored result for image " + imageName)
        
        #base_positions = numpy.matrix([[p[0], p[1]] for p in base_positions])
    for imagePath in images_paths:  #for each image path
        detector = FacialLandmarkDetector(imagePath)    #create FacialLandmarkDetector object for image
        if warp == True:
            detector.saveImage(destination, additional_name = "_non_warped")
            detector.warpe_image(base_positions)
        
        positions = detector.detectFacialLandmarks(draw=True, normalize=True)    # get positions for image with normalization set to True
        landmarksPositions.append(positions)
        if store == True:   #is True, save image to destination folder
            detector.saveImage(destination, additional_name = "_warped")
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
    
#Method returns image path for given name of image (000,001 ...)
def getImagePath(database_folder, imageName):
    loader = DatabaseLoaderXMVTS2(database_folder)
    imagePath = loader.imagePathFinder(imageName, "")
    return imagePath


if __name__ == "__main__":

    templates_database = "/home/matej/Diplomski/baze/Templates/baza_templates"
    templates_similarity_test = "/home/matej/Diplomski/baze/Templates/templates_similarity_test"

    templates_destination = "/home/matej/Diplomski/baze/Templates/KazemiTemplates"

    findFacialLandmarksOnTemplateImages(templates_database, templates_destination, warp = True, showImages = False, store=True, storePositions = True)


