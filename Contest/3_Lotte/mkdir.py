### 125 caterogories per model * 8 models = 1000 categories

import os, shutil

# path 지정해주고 폴더 만들기
original_dataset_dir = 'Path'

base_dir = 'path'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

# 파일 옮기기
# 처음 1000개의 고양이 이미지를 복사합니다
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dir, fname)
    shutil.copyfile(src, dst)

