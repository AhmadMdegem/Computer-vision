import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.ops.confusion_matrix import confusion_matrix

X_train = []
y_train = []
z = []
directory = 'output\\'+'train\\'

for i in range(0, 27):
    directory = 'output\\'+'train\\' + str(i)
    for str2 in os.listdir(directory):
        filename = 'output\\'+'train\\' + str(i) + '\\' + str2
        img = cv.imread(filename, 0)
        X_train.append(img.flatten())
        y_train.append(i)
    z.append(i)


for i in range(0, len(X_train)):
    X_train[i] = np.array(X_train[i]).reshape(32, 32)
y_train = np.array(y_train)

X_train = tf.keras.utils.normalize(X_train, axis=1)
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
    z.append(i)

for i in range(0, len(X_val)):
    X_val[i] = np.array(X_val[i]).reshape(32, 32)
y_val = np.array(y_val)
X_val = tf.keras.utils.normalize(X_val, axis=1)

#######################################################
# Configurations
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32, 32)))

# # 1- without regularization accuracy = 0.727822
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))


# # 2.1- with regularization L1 lamda = 0.01 accuracy = 0.0423
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(0.01)))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(0.01)))

# # 2.2- with regularization L1 lamda = 0.001 accuracy = 0.66129
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(0.001)))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L1(0.001)))

# # 3.1- with regularization L2 lamda = 0.01  accuracy = 0.67540
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.01)))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.01)))

# # 3.2- with regularization L2 lamda = 0.001  accuracy = 0.725806
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.001)))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.001)))

# 4- with Dropout p = 0.5  accuracy = 0.729838 max
model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
model.add(Dropout(0.5))

# # 5.1- with regularization L2 lamda = 0.01  & Dropout p = 0.5  accuracy = 0.65322
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.01)))
# model.add(Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.01)))
# model.add(Dropout(0.5))

# # 5.2- with regularization L2 lamda = 0.001 && Dropout p = 0.5  accuracy = 0.71975809
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.001)))
# model.add(Dropout(0.5))
# model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_regularizer=regularizers.L2(0.001)))
# model.add(Dropout(0.5))

model.add(tf.keras.layers.Dense(units=27, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
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


confusion_matrix(y2, X_pred)

#########################################################
print('letter'+'     ' + 'accuracy')
for i in range(0, 27):
    print(str(i)+'        '+str(letter_accuracy[i]))
print("average :   " + str(sum(letter_accuracy)/len(letter_accuracy)))
##########################################################