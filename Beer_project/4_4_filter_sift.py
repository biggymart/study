import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from statistics import mean
import cv2
import os

def get_dir(beer):
    std_img_dir = 'C:/data/image/beer/{0}/frame0.jpg'.format(beer)

    test_base_dir = 'C:/data/image/beer_selenium/{0}/'.format(beer)
    test_img_lst = os.listdir(test_base_dir)
    return std_img_dir, test_base_dir, test_img_lst

def feature_match(img1_dir, img2_dir):
    img1 = cv2.imread(img1_dir)  
    img2 = cv2.imread(img2_dir) 
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    return len(matches)

def mk_bucket(beer):
    bucket = []
    dir1, dir2, dir3 = get_dir(beer)
    for i in range(len(dir3)):
        num_feature = feature_match(dir1, dir2 + dir3[i])
        bucket.append(num_feature)
    return bucket

def rmv_blw_avg(bucket, beer):
    dir0, dir1, dir2 = get_dir(beer)
    count = 0

    print(beer, int(mean(bucket)), "못 넘으면 죽는다")
    # 명부 작성이요
    idx = []
    for i, v in enumerate(bucket):
        if v < mean(bucket):
            idx.append(i)
            count += 1

    # 형 집행이요
    idx_rev = reversed(idx)
    for i in idx_rev:
        os.remove(dir1 + dir2[i])
    print(beer, count, "files deleted")

bucket_cass = mk_bucket('cass')
bucket_filgood = mk_bucket('filgood')
bucket_filite = mk_bucket('filite')
bucket_hite = mk_bucket('hite')

rmv_blw_avg(bucket_cass, 'cass')
rmv_blw_avg(bucket_filgood, 'filgood')
rmv_blw_avg(bucket_filite, 'filite')
rmv_blw_avg(bucket_hite, 'hite')

# https://m.blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220655420471&proxyReferer=https:%2F%2Fwww.google.com%2F