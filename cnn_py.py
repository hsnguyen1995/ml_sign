import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

model = load_model("cnn_sign.h5")

test_data_dir = "data/test/"

def load_testing_data():
    img_size = 64
    testing_data = []
    labels = []
    for image in os.listdir(test_data_dir):
        try:
            img_array = cv2.imread(test_data_dir + image)
            new_array = cv2.resize(img_array, (img_size,img_size))
            testing_data.append(new_array)
            labels.append(image)
        except Exception as e:
            pass
            
    testing_data_arr = np.array(testing_data)
    testing_data_norm = testing_data_arr.astype('float32')/255
#     print(labels)
#     labels_categ = to_categorical(labels)
    return testing_data_norm, labels
# , labels_categ
#     X_test is testing_data_norm
#     y_test is labels_categ
testing_data_norm, labels_categ = load_testing_data()

predictions = [model.predict_classes(image.reshape(1,64,64,3))[0] for image in testing_data_norm]

letter_decode = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25}

def decodePredictions(p):
    decoded_predictions = []
    for i in range(len(p)):
        for letter in letter_decode:
            if p[i] == letter_decode[letter]:
                decoded_predictions.append(letter)
    return decoded_predictions

prediction_letters = decodePredictions(predictions)
print(prediction_letters)