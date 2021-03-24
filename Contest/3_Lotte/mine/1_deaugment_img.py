# Back to the original
# 증폭을 했는데 다시 되돌리고 싶을 때 쓰는 파일
# 즉, 48개의 이미지만 제외하고 다 제거하고 싶을 때 사용

from natsort import natsorted
import os

TRAIN_DIR = 'C:/data/LPD_competition/train'
train_fnames = natsorted(os.listdir(TRAIN_DIR))

original_img = ['0.jpg','1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg',\
              '10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg','16.jpg','17.jpg','18.jpg','19.jpg',\
              '20.jpg','21.jpg','22.jpg','23.jpg','24.jpg','25.jpg','26.jpg','27.jpg','28.jpg','29.jpg',\
              '30.jpg','31.jpg','32.jpg','33.jpg','34.jpg','35.jpg','36.jpg','37.jpg','38.jpg','39.jpg',\
              '40.jpg','41.jpg','42.jpg','43.jpg','44.jpg','45.jpg','46.jpg','47.jpg']

for idx, folder in enumerate(train_fnames): # 1000개의 폴더에 대하여
    # if idx >= 1: # 폴더 1개만 시험삼아 해보기
    #     break

    base_dir = TRAIN_DIR + '/' + folder + '/' # 'C:/data/LPD_competition/train/0/'
    img_lst = natsorted(os.listdir(base_dir))
    print("deleting", idx, "th folder...")
    for img in img_lst:
        if img not in original_img:
            os.remove(base_dir + str(img))