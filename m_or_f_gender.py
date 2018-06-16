import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam

from skimage.io import imread
from skimage.transform import resize


class M_F_Detector:
    
    def __init__(self, model_path):


        self.model = Sequential()
        self.model.add(Conv2D(32, (5, 5), input_shape = (64,64,3), activation='relu'))
        self.model.add(Conv2D(32, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1,activation='sigmoid'))

        self.model.load_weights(model_path)

        self.model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.0001, beta_1=0.95, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics = ['accuracy'])

    def predict(self, image_path):
        """ Takes as input - classification model(m),path to image file location(img_file)
        Returns - Probability of image class as output by model m as a tuple."""
        im = imread(image_path)
        im = resize(im,self.model.input_shape[1:3],mode='reflect')

        im = im[np.newaxis,:,:,:]

        res = self.model.predict_proba(im,verbose=0)

        if res[0][0] >= 0.5: #man
            return 0
        else:   #woman
            return 1
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


    model_path = "gender_models/generator_model.hdf5"

    detector = M_F_Detector(model_path)

    predicted = detector.predict(imagePath)
    print(predicted)

    label = detector.predict_label(imagePath)

    print(label)
