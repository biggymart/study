import numpy as np
import cv2

path_to_img = 'C:/Users/snu20/Desktop/cass.jpg'

def SIFT():
    img = cv2.imread(path_to_img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2, img3 = None, None

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(imgray, None)
    print(len(kp), len(des))

    img2 = cv2.drawKeypoints(imgray, kp, img2)
    img3 = cv2.drawKeypoints(imgray, kp, img3,
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    cv2.imshow('SIFT1', img2)
    cv2.imshow('SIFT2', img3)
    cv2.imwrite('C:/Users/snu20/Desktop/sift1.jpg', img2)
    cv2.imwrite('C:/Users/snu20/Desktop/sift2.jpg', img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

SIFT()