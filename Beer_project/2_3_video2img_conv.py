# src: https://shilan.tistory.com/entry/Python%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%99%EC%98%81%EC%83%81%EC%9C%BC%EB%A1%9C%EB%B6%80%ED%84%B0-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%B6%94%EC%B6%9C-Pythonv27-OpenCV-Windows

import cv2
import os

# feel_good, filite_fresh 완료
# cass, cass_light, filite, hite 해야 함
src_dir = 'C:/data/video/'
tar_dir = 'C:/data/image/beer/'
beer_name = 'hite'
video_file = '{0}.mp4'.format(beer_name)
# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
vidcap = cv2.VideoCapture(src_dir + video_file)

count = 0
'''
# 방법1
while(vidcap.isOpened()):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()
 
    # 캡쳐된 이미지를 저장하는 함수 
    cv2.imwrite("C:/data/image/beer/filite/frame%d.jpg" % count, image)
 
    print('Saved frame%d.jpg' % count)
    count += 1
 
vidcap.release()
'''

if not os.path.exists(tar_dir + beer_name):
    os.mkdir(tar_dir + beer_name)


# 방법2
while(vidcap.isOpened()):
    ret, image = vidcap.read()
 
    if(int(vidcap.get(1)) % 20 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        # cv2.imwrite("C:/data/image/beer/feel_good/frame%d.jpg" % count, image)
        cv2.imwrite("{0}{1}/frame{2}.jpg".format(tar_dir, beer_name, count), image)
        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()
