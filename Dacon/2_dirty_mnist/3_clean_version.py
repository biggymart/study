import os
import cv2 as cv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train_base_dir = 'C:/data/mnist/dirty_mnist_2nd/'
test_base_dir = 'C:/data/mnist/test_dirty_mnist_2nd/'

train_file_lst = os.listdir(train_base_dir)
test_file_lst = os.listdir(test_base_dir)

img = cv.imread(train_base_dir + train_file_lst[0], flags=cv.IMREAD_GRAYSCALE)
train_label = pd.read_csv('C:/data/mnist/dirty_mnist_2nd_answer.csv', index_col='index')

kernel = np.ones((3, 3), np.uint8)
erosion = cv.erode(img, kernel, iterations = 1)
ret, thresh = cv.threshold(erosion, 100, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.imshow('Contours', thresh)
cv.waitKey(0)
cv.destroyAllWindows()