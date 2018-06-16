
from gender_detection_keras import *
from unified_face_landmar_gender_recognition import *
from m_or_f_gender import *



class VotingGenderDetector:
    def __init__(self, model1_path, model2_path, model3_path):
        self.model1 = GenderDetectKeras(model1_path)
        self.model2 = UnifiedGenderDetection(model2_path)
        self.model3 = M_F_Detector(model3_path)
        
    def predict(self, image_path):
        predicted1 = self.model1.predict(image_path)
        predicted2 = self.model2.predict(image_path)
        
        if predicted1 == predicted2:
            return predicted1
        else:
            predicted3 = self.model3.predict(image_path)
            return predicted3
            
    def predict_label(self, image_path):
        predicted = self.predict(image_path)
        if (predicted == 0):
            return "man"
        else:
            return "woman"
            
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

    imagePath = imagePath_same3 

    model1_path = "gender_models/gender_detection_keras.model"
    model2_path = "gender_models/svm_model.pkl"
    model3_path = "gender_models/generator_model.hdf5"

    detector = VotingGenderDetector(model1_path, model2_path, model3_path)

    predicted = detector.predict(imagePath)
    print(predicted)

    label = detector.predict_label(imagePath)

    print(label)
