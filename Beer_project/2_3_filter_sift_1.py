import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import cv2 
import matplotlib.pyplot as plt
import os

# SIFT를 활용하여 cass한 번 해보자
# X 재료
THRESH_HOLD = 125
beer_lst = os.listdir('C:/data/image/beer/') # ['cass', 'filgood', 'filite', 'hite']

# X: 맥주의 이름, y: 해당 맥주의 디렉토리 3가지
def get_dir(beer):
    # archetypical image of beer can (첫번째 사진)
    std_img_dir = 'C:/data/image/beer/{0}/frame0.jpg'.format(beer)

    # data from selenium
    test_base_dir = 'C:/data/image/beer_selenium/{0}/'.format(beer)
    test_img_lst = os.listdir(test_base_dir)
    return std_img_dir, test_base_dir, test_img_lst

# X: 이미지 두 개의 디렉토리 y: 두 이미지의 매칭 갯수
def feature_match(img1_dir, img2_dir):
    img1 = cv2.imread(img1_dir)  
    img2 = cv2.imread(img2_dir) 

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #sift
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    return len(matches)

# list_attach
bucket_cass = []
cass_dir1, cass_dir2, cass_dir3 = get_dir('cass')
for i in range(len(cass_dir3)):
    num_feature = feature_match(cass_dir1, cass_dir2 + cass_dir3[i])
    bucket_cass.append(num_feature)
    print(i, "attached")
print(bucket_cass)

bucket_filgood = []
filgood_dir1, filgood_dir2, filgood_dir3 = get_dir('filgood')
for i in range(len(filgood_dir3)):
    num_feature = feature_match(filgood_dir1, filgood_dir2 + filgood_dir3[i])
    bucket_filgood.append(num_feature)
    print(i, "attached")
print(bucket_filgood)

bucket_filite = []
filite_dir1, filite_dir2, filite_dir3 = get_dir('filite')
for i in range(len(filite_dir3)):
    num_feature = feature_match(filite_dir1, filite_dir2 + filite_dir3[i])
    bucket_filite.append(num_feature)
    print(i, "attached")
print(bucket_filite)

bucket_hite = []
hite_dir1, hite_dir2, hite_dir3 = get_dir('hite')
for i in range(len(hite_dir3)):
    num_feature = feature_match(hite_dir1, hite_dir2 + hite_dir3[i])
    bucket_hite.append(num_feature)
    print(i, "attached")
print(bucket_hite)

# statistics
from statistics import *
print(quantiles(bucket_cass, n=4))
print(quantiles(bucket_filgood, n=4))
print(quantiles(bucket_filite, n=4))
print(quantiles(bucket_hite, n=4))

'''
# remove that did not pass
for i, v in enumerate(bucket_cass):
    if v < THRESH_HOLD:
        os.remove(cass_dir2 + cass_dir3[i])
        print(cass_dir2 + str(i) + " file removed")

for i, v in enumerate(bucket_filgood):
    if v < THRESH_HOLD:
        os.remove(filgood_dir2 + filgood_dir3[i])
        print(filgood_dir2 + str(i) + " file removed")

for i, v in enumerate(bucket_filite):
    if v < THRESH_HOLD:
        os.remove(filite_dir2 + filite_dir3[i])
        print(filite_dir2 + str(i) + " file removed")

for i, v in enumerate(bucket_hite):
    if v < THRESH_HOLD:
        os.remove(hite_dir2 + hite_dir3[i])
        print(hite_dir2 + str(i) + " file removed")
'''


