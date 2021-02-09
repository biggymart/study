import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 이미지를 데이터로 전처리하는 것 (증폭, 변환)
# OpenCV, loadimage 있는데 그 중 하나일 뿐

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' # 최근접기법
)
# 현재 어떤 이미지도 로딩하지 않았고
# 이런 훈련과 테스트 셋을 준비하겠다고 단순히 선언한 것일 뿐 
# (이렇게 변환하겠다, 함수 설정과 비슷)

test_datagen = ImageDataGenerator(rescale=1./255)
# 훈련할 때는 충분히 많이 훈련하는 게 좋겠지만
# 시험 잘 보려고 하는데 진짜 시험 문제를 증폭시키고 싶니?

# flow 혹은 flow_from_directory (데이터로 만들어준다, 폴더 구조로 레이블해줄 수 있음)

# train_generator
# 이미지의 데이터화 (fit과 유사)
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train', 
    # flow의 경우 데이터를 넣어주면 됨 
    # ad와 normal 폴더 안에 있는 80개 이미지 전체를 데이터화
    target_size=(150, 150),
    color_mode='grayscale',
    classes=['ad', 'normal'],
    # 임의의 사이즈 설정해줌 shape=(80, 150, 150, 1)
    batch_size=5, 
    # xy_train[0]을 print해보면 x가 5, y가 5 들어가 있는 걸 볼 수 있다 
    # xy_train[0][0]은 x부분, xy_train[0][1]은 부분
    # xy_train[0][0].shape은 (5, 150, 150, 1)
    # xy_train[0][1].shape은 (5,)
    # xy_train[0][1]은 0과 1로 구성됨
    class_mode='binary'
) # Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150, 150),
    color_mode='grayscale',
    classes=['ad', 'normal'],
    batch_size=5,
    class_mode='binary'
) # Found 120 images belonging to 2 classes.

# 생성이 되었는지 확인해보자
print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001FA9BB18550>
# sklearn.dataset도 이런 구조로 들어가 있지 (x, y가 함께 한 군데로 저장)

print(xy_train[0])
# 첫번째 batch

### 아직 증폭 안 한 상태임 ###
# fit_generator는 batch_size가 딱 떨어질 필요 없이도 잘 돌아감