# https://ko.ojit.com/so/python/1590982
import numpy as np
import cv2
from matplotlib import pyplot as plt


# FIXME: doesn't work
def deskew():
    im_out = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    plt.imshow(im_out, 'gray')
    plt.show()


# resizing images to improve speed
factor = 0.4
img1 = cv2.resize(cv2.imread("image.png", 0), None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(cv2.imread("imageSkewed.png", 0), None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                          ]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                          ]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    deskew()

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

# show matching keypoints
draw_params = dict(matchColor=(0, 255, 0),  # draw  matches in  green   color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only    inliers
                   flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray')
plt.show()