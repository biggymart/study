# import the necessary packages
# 트리계열로 해봐야하나

import imutils
import cv2
import os

######
TRAIN_DIR = 'C:/data/LPD_competition/train/'



# load the image, convert it to grayscale, and blur it slightly


for f in range(len(os.listdir(TRAIN_DIR))): # train에 들어있는 1000개의 폴더 개수만큼 반복
    img_lst = os.listdir(TRAIN_DIR + str(f))

    # 마지막 10개 이미지만 해볼까
    for i in range(len(img_lst)): # 각 폴더 안의 이미지 개수만큼 반복
        each_img = TRAIN_DIR + str(i) + '/' + img_lst[i]
        image = cv2.imread(each_img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cv2.imwrite(img)

# cv2.imwrite(TRAIN_DIR + 'train/' + '0/' + 'thresh.jpg',thresh)


# # https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/