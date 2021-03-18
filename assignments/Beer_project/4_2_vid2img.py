import cv2
import os

src_dir = 'C:/data/video/'
tar_dir = 'C:/data/image/beer/'
beer_name = 'hite'
video_file = '{0}.mp4'.format(beer_name)

vidcap = cv2.VideoCapture(src_dir + video_file)

while(vidcap.isOpened()):
    ret, image = vidcap.read()
    count = 0
    if(int(vidcap.get(1)) % 20 == 0):
        cv2.imwrite("{0}{1}/frame{2}.jpg".format(tar_dir, beer_name, count), image)
        count += 1

vidcap.release()
