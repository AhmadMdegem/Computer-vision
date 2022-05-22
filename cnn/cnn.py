import os
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Flatten
from numpy import newaxis
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.ops.confusion_matrix import confusion_matrix

X_train = []
y_train = []

directory = 'output\\'+'train\\'

for i in range(0, 27):
    directory = 'output\\'+'train\\' + str(i)
    for str2 in os.listdir(directory):
        filename = 'output\\'+'train\\' + str(i) + '\\' + str2
        img = cv.imread(filename, 0)
        X_train.append(img.flatten())
        y_train.append(i)



for i in range(0, len(X_train)):
    X_train[i] = np.array(X_train[i]).reshape(32, 32)
y_train = np.array(y_train)

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_train = np.array(X_train)
#######################################################
X_val = []
y_val = []

for i in range(0, 27):
    directory = 'output\\'+'val\\' + str(i)
    for str2 in os.listdir(directory):
        filename = 'output\\'+'val\\' + str(i) + '\\' + str2
        img = cv.imread(filename, 0)
        X_val.append(img.flatten())
        y_val.append(i)


for i in range(0, len(X_val)):
    X_val[i] = np.array(X_val[i]).reshape(32, 32)
y_val = np.array(y_val)
X_val = tf.keras.utils.normalize(X_val, axis=1)

#######################################################
# Configurations
model = tf.keras.models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))


model.add(layers.Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))


model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))


model.add(Dense(27,activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X_train = np.reshape(X_train,(4050,32,32))
x = X_train[:, :, :, newaxis]
# accuracy with == 0.8245967626571655
# accucracy without == 0.8346773982048035
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=False,vertical_flip=False,rotation_range=10,shear_range=0.2,brightness_range=(0.2, 1.8),rescale=1. / 255)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

loss, accuracy = model.evaluate(X_val, y_val)
print(accuracy)
###################################################
# graph plt
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, 51)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


######################################################
letter_accuracy = []
y2 = []
X_pred = []
for item in range(0, 27):
    x_le = []
    y_le = []

    for item1 in os.listdir("output\\test\\"+str(item)):
        img = cv.imread("output\\test\\" + str(item) + "\\" + item1)[:, :, 0]
        x_le.append(img)
        img = np.array([img])
        y2.append(item)
        y_le.append(item)
        y_pred = model.predict(img)
        X_pred.append(np.argmax(y_pred))
    y_le = np.array(y_le)
    x_le = tf.keras.utils.normalize(x_le, axis=1)
    loss, accuracy = model.evaluate(x_le, y_le)
    letter_accuracy.append(accuracy)


print(confusion_matrix(y2, X_pred))

#########################################################
print('letter'+'     ' + 'accuracy')
for i in range(0, 27):
    print(str(i)+'        '+str(letter_accuracy[i]))
print("average :   " + str(sum(letter_accuracy)/len(letter_accuracy)))
##########################################################