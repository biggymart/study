import os
import imutils
import cv2 as cv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train_base_dir = 'C:/data/mnist/dirty_mnist_2nd/'
test_base_dir = 'C:/data/mnist/test_dirty_mnist_2nd/'
train_file_lst = os.listdir(train_base_dir)
test_file_lst = os.listdir(test_base_dir)

img = cv.imread(train_base_dir + train_file_lst[0])
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('original', img)

# blur = cv.medianBlur(img, 5)
# blur = cv.blur(img, (3,3))
# blur = cv.GaussianBlur(img,(5,5),0)
blur = cv.bilateralFilter(img,9,75,75)

cv.imshow('blur', blur)
# train_label = pd.read_csv('C:/data/mnist/dirty_mnist_2nd_answer.csv', index_col='index')

kernel = np.ones((3, 3), np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
cv.imshow('erosion', erosion)

ret, thresh = cv.threshold(erosion, 100, 255, cv.THRESH_BINARY)
cv.imshow('thresh', thresh)

# cv.imwrite()
##################################################################################################

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.RETR_TREE, cv.RETR_LIST
'''
for i, contour in enumerate(contours):
     x, y, w, h = cv.boundingRect(contour)
     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
'''
cv.drawContours(thresh, contours, -1, (0,255,0), 3)
print("Number of contours :", str(len(contours)))

cv.imshow('Contours', thresh)
cv.waitKey(0)
cv.destroyAllWindows()
