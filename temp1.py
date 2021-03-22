from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
import os
from natsort import natsorted

# 10개의 판별 모델을 만들어서 competition 벌이는 대환장 소스코드!


atom = 100

TRAIN_DIR = 'C:/data/LPD_competition/train'
TEST_DIR = 'C:/data/LPD_competition/test'
model_path = 'C:/data/modelCheckpoint/lotte_0318_1_{epoch:02d}-{val_loss:.4f}.hdf5'

train_fnames = os.listdir(TRAIN_DIR)
train_fnames = natsorted(train_fnames)
print(train_fnames[0:10])

# 1000의 약수인지 확인해주는 함수
def test_atom(atom):
    if 1000 % atom == 0:
        print("atom well set!")
    else: 
        raise ValueError('atom should evenly divide 1000')
test_atom(atom)


train_datagen = ImageDataGenerator(
    validation_split = 0.2,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    horizontal_flip=True,
    rotation_range=5,
    zoom_range=0.2,
    preprocessing_function = preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

for i in range(1000/atom):
    train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        class_mode='categorical',
        classes=train_fnames
    )
