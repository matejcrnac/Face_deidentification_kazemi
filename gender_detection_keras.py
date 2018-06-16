#Classifier 1-------------------------------------------------------------

# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv



class GenderDetectKeras:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    # 0 - man
    # 1 - woman
    def predict(self, image_path, confidence = False):
        # read input image
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read input image")
            exit()
        
        #preproces image
        # detect faces in the image
        face, confidence = cv.detect_face(image)


         # get corner points of face rectangle      
        face = face[0] 
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = self.model.predict(face_crop)[0]

        # get label with max accuracy
        idx = np.argmax(conf)
        
        if confidence == True:
            return conf
        return idx
        
    def predict_label(self, image_path):
        classes = ['man','woman']
        predicted = self.predict(image_path)
        label = classes[predicted]
        
        return label
    
if __name__ == "__main__":

    imagePath_same1 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/000/000_1_1.ppm" # sve dobro,
    imagePath_same2 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/001/001_1_1.ppm" # sve dobro
    imagePath_same3 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/002/002_1_1.ppm" # spec bolji (k4 prob)
    imagePath_same4 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/004/004_1_1.ppm" # spec bolji od all dist
    imagePath_same5 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/005/005_1_1.ppm" # spec najbolji
    imagePath_same6 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/006/006_1_1.ppm"
    imagePath_same7 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/007/007_1_1.ppm" # nije dobar
    imagePath_same8 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/008/008_1_1.ppm"
    imagePath_same9 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/009/009_1_1.ppm"
    imagePath_same10 = "/home/matej/Diplomski/baze/baze_original/baza_XMVTS2/010/010_1_1.ppm"


    model_path = "gender_models/gender_detection_keras.model"

    imagePath = imagePath_same3
                         

    classifier = GenderDetectKeras(model_path)

    predicted = classifier.predict(imagePath)

    print(predicted)

    label = classifier.predict_label(imagePath)
    
    print("Detected gender is:")
    print(label)
