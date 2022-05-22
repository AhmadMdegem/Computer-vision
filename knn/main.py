import os

import cv2
import matplotlib.pyplot as plt
import numpy as np




def padding_resize():
    directory1 = 'hhd_dataset'
    for namee in os.listdir(directory1):
        #plus padding
        directory = 'hhd_dataset\\'+namee
        for str1 in os.listdir(directory):
            filename = 'hhd_dataset\\'+namee+'\\'+str1
            img = cv2.imread(filename, 0)
            w , h = img.shape

            if w > h:
                h1 = w - h ;
                img1 = cv2.copyMakeBorder(img, 0 ,0 , int(h1/2), int(h1/2), cv2.BORDER_CONSTANT, None, value=255)
                cv2.imwrite('hhd_dataset_after\\'+namee+'\\'+str1, img1)
            if w < h:
                w1 = h - w;
                img1 = cv2.copyMakeBorder(img,int(w1 / 2), int(w1 / 2),0, 0, cv2.BORDER_CONSTANT, None, value=255)
                cv2.imwrite('hhd_dataset_after\\'+namee +'\\'+str1, img1)
            if w == h:
                cv2.imwrite('hhd_dataset_after\\'+namee +'\\'+str1, img1)

        # resize to 32 32
        directory = 'hhd_dataset_after\\'+namee
        for str1 in os.listdir(directory):
            filename = 'hhd_dataset_after\\'+namee +'\\' + str1
            img = cv2.imread(filename, 0)
            newsize = (32, 32)
            img1 = cv2.resize(img,newsize)
            cv2.imwrite('hhd_dataset_finale\\'+namee +'\\' + str1, img1)



def main():
    # padding_resize()
    print('hi')


if __name__ == "__main__":
    main()