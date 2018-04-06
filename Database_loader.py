import os
import zipfile

class DatabaseLoaderXMVTS2:

    def __init__(self, database_folder):
        self.database_folder = database_folder

    def loadDatabase(self, extension, imageNum = ""):
        images_paths = []
        if imageNum != "":
            imageNum = "_"+imageNum
        for root, dirs, files in os.walk(self.database_folder):
            for file in files:
                if file.endswith(extension) and imageNum + "_1" in file :
                    fName = os.path.join(root, file)
                    images_paths.append(fName)

        return sorted(images_paths)

    def extractDatabase(self, extension, goalFolder):
        for root, dirs, files in os.walk(self.database_folder):
            for file in files:
                if file.endswith(extension):
                    fName = os.path.join(root, file)
                    zip_ref = zipfile.ZipFile(fName, 'r')
                    zip_ref.extractall(goalFolder)
                    zip_ref.close()
                    print("File " + file + " extracted.")
        print("Extracting finished")

    def imagePathFinder(self, imageName, imageNumber):
        if imageNumber != "":
            imageNumber = "_"+imageNumber
        for root, dirs, files in os.walk(self.database_folder):
            for file in files:
                if imageName in file and imageNumber + "_1" in file:
                    fName = os.path.join(root, file)
                    return fName
    def getImagesPath(self, imageNameList, imageNumber):
        images_paths = []
        if imageNumber != "":
            imageNumber = "_"+imageNumber
        for root, dirs, files in os.walk(self.database_folder):
            for file in files:
                for imageName in imageNameList:
                    if imageName in file and imageNumber + "_1" in file:
                        fName = os.path.join(root, file)
                        images_paths.append(fName)
                        break
        return images_paths
if __name__ == "__main__":
    #database_folder = "/media/matej/D/Databases/Faces/XMVTS2/baza/full_database_v1"
    database_folder = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_XMVTS2"
    loader = DatabaseLoaderXMVTS2(database_folder)
    images_paths = loader.loadDatabase('ppm')
    print(images_paths)
    #loader.extractDatabase("zip", "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_XMVTS2")
    imagePath = loader.imagePathFinder("083", "1")
    print(imagePath)

    imagePaths = loader.getImagesPath(["000", "034", "044"], "1")
    print(imagePaths)
