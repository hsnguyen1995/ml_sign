import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical



# ------------------Loading training data starts here------------------
training_data_dir = "data/train"
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

img_size = 64
training_data = []
labels = []


def create_training_dataset():
    for letter in letters:
        path = os.path.join(training_data_dir, letter)
        label_class = letters.index(letter)
        for image in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,image))
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_data.append(new_array)
                labels.append(label_class)
            except Exception as e:
                pass
            
    training_data_arr = np.array(training_data)
    training_data_norm = training_data_arr.astype('float32')/255
    
    labels_categ = to_categorical(labels)
    print(len(training_data_norm), training_data_norm.shape)
    return training_data_norm, labels_categ
#     X_train is training_data_norm
#     y_train is labels_categ

training_data_norm, labels_categ = create_training_dataset()



# ------------------CNN Building starts here------------------
model = Sequential()

model.add(Conv2D(16, (3,3), activation="relu", padding="same", input_shape = training_data_norm.shape[1:]))
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(Conv2D(64, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Conv2D(128, (3,3), activation="relu", padding="same"))
model.add(Conv2D(256, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(26, activation="softmax"))
          
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(training_data_norm, labels_categ, batch_size=64, epochs=15, validation_split=0.1)


# ------------------Save the model------------------
model.save("cnn_sign.h5") 



# ------------------Path to Test Directory and pull data------------------
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

    return testing_data_norm, labels

testing_data_norm, labels_categ = load_testing_data()


# ------------------Pass testing data through the model to produce predictions------------------
predictions = [model.predict_classes(image.reshape(1,64,64,3))[0] for image in testing_data_norm]
print(predictions)