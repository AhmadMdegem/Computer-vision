import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier




##############################################train
X=[]
y=[]


directory = 'output\\'+'train\\'

for str1 in os.listdir(directory):
    directory = 'output\\'+'train\\' + str1
    for str2 in os.listdir(directory):
        filename = 'output\\'+'train\\' + str1 + '\\' + str2
        img = cv2.imread(filename, 0)
        X.append(img.flatten())
        y.append(str1)



knn= KNeighborsClassifier(n_neighbors=3)
X = np.array(X)
knn.fit(X,y)

############################val & true k
X1=[]
y1=[]
letter_accuracy=[]
directory = 'output\\'+'val\\'

for str1 in os.listdir(directory):
    directory = 'output\\'+'val\\' + str1
    x_le = []
    y_le = []
    for str2 in os.listdir(directory):
        filename = 'output\\'+'val\\' + str1 + '\\' + str2
        img = cv2.imread(filename, 0)
        X1.append(img.flatten())
        y1.append(str1)
        x_le.append(img.flatten())
        y_le.append(str1)
        letter_accuracy.append(knn.score(x_le, y_le))

max=0.0
X1 = np.array(X1)
for k in range(1,16,2):
    if k!=1:
        knn= KNeighborsClassifier(n_neighbors=k)
        knn.fit(X1,y1)
        oo=knn.score(X1, y1)
        if max < oo:
            max = knn.score(X1, y1)
            #print(max)

#################################################

print('letter'+'     '+ 'accuracy')
for i in range(0,27):
    print(str(i)+'        '+str(letter_accuracy[i]))

#################################################test

X2=[]
y2=[]
directory = 'output\\'+'test\\'

for str1 in os.listdir(directory):
    directory = 'output\\'+'test\\' + str1
    for str2 in os.listdir(directory):
        filename = 'output\\'+'test\\' + str1 + '\\' + str2
        img = cv2.imread(filename, 0)
        X2.append(img.flatten())
        y2.append(str1)
X2 = np.array(X2)
y_pred = knn.predict(X2)

#print(confusion_matrix(y2,y_pred))


