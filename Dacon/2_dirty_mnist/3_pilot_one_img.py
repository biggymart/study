### Description ###
# 이미지 한 장에 대해서 전처리하도록 해보죠. 잘 되면 더 확장하도록 해요.
# Date: 2021-02-25


### 0. 라이브러리는 여기에 정리하도록 해요
import os
import cv2 as cv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Overall Workflow
# Input -> Grayscale -> Morph Gradient -> Adaptive Threshold -> Contour 추출 -> resize
# 모델링 이후 분류...


### 1. 가장 먼저, 디렉토리를 정리해봅시다!
#1-1. 베이스 디렉토리
train_base_dir = 'C:/data/mnist/dirty_mnist_2nd/'
test_base_dir = 'C:/data/mnist/test_dirty_mnist_2nd/'

#1-2. 파일명 리스트로 정리 (나중에 반복문을 위해)
train_file_lst = os.listdir(train_base_dir)
test_file_lst = os.listdir(test_base_dir)


### 2. 정리한 디렉토리를 바탕으로 이미지를 한 장만 긁어와 볼까요?
img = cv.imread(train_base_dir + train_file_lst[0], flags=cv.IMREAD_GRAYSCALE)
# img_test = cv.imread(test_base_dir + test_file_lst[0], flags=cv.IMREAD_GRAYSCALE)
# 가져온 이미지를 확인해봐요
# print(img_train.shape) # (256, 256)
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 이미지 크기랑 생긴 건 알겠어요. 근데 여기에 도대체 뭐가 들어있다는 거죠?
# 월간 데이콘 7의 글자 10 ~ 14개가 무작위로 들어가 있데요. 근데 뭐가 들어가 있는 건지는 csv 파일을 확인해봐야할 것 같아요
# 즉, csv파일 = X(혹은 input)의 레이블이군요! 첫번째 데이터에 대해서 레이블을 확인해볼게요.
train_label = pd.read_csv('C:/data/mnist/dirty_mnist_2nd_answer.csv', index_col='index')
# print(train_label)
# print(train_label.shape) # (50000, 27)

# 사진 속에 어떤 글자가 있는지 알아보기 위해선 value가 1인 column이 무엇인지 알아야 해요
# print(train_label.loc[0])
# Y (label)은 나중에 구성하도록 하죠


### 미친 전처리리리리릴리리리리리릴리ㅣ리ㅣ리리리리ㅣㄹ 고고고ㅗ고고고고고고고곡
# 일단은 X의 전처리를 해보도록 해요.

# Morph Gradient (erosion, dilation, opening, closing, gradient, blackhat)
kernel = np.ones((3, 3), np.uint8)
erosion = cv.erode(img, kernel, iterations = 1)
# cv.imshow('Erosion', erosion)

# Adaptative Threshold (Global Threshold, Gaussian, Mean)
ret, thresh = cv.threshold(erosion, 100, 255, cv.THRESH_BINARY) # cv.THRESH_BINARY cv.THRESH_OTSU cv.THRESH_TOZERO

# cv.imshow('global threshold', thresh)
# cv.waitKey(0)
# cv.destroyAllWindows()

# import sys
# np.set_printoptions(threshold=sys.maxsize)
# print(thresh)


# Contour Extraction
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print("Number of Contours :", str(len(contours))) # 395

# cv.drawContours(thresh, contours, -1, (0,255,0), 3)
# cv.drawContours(img, contours, 3, (0,255,0), 3)
# cnt = contours[4]
# cv.drawContours(img, [cnt], 0, (0,255,0), 3)

# cv.drawContours(img, contours, cv.CHAIN_APPROX_SIMPLE, (0,255,0), 3)

cv.imshow('Contours', thresh)
cv.waitKey(0)
cv.destroyAllWindows()


### 참고문헌
# 딥러닝과 OpenCV를 활용하여 사진 속 글자 검출하기
# https://d2.naver.com/helloworld/8344782
# OpenCV 기초
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
# OpenCV Mophology
# https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
# Adaptative Threshold
# https://hoony-gunputer.tistory.com/entry/OpenCV-python-adaptive-Threshold
# Threshold types
# https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576
# Contours
# https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html