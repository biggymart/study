import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator
# 이미지의 데이터화 (fit과 유사)
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train', 
    target_size=(150, 150),
    color_mode='grayscale',
    classes=['ad', 'normal'],
    batch_size=160, 
    class_mode='binary'
) # Found 160 images belonging to 2 classes.

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150, 150),
    color_mode='grayscale',
    classes=['ad', 'normal'],
    batch_size=120,
    class_mode='binary'
) # Found 120 images belonging to 2 classes.

np.save('../data/image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/npy/keras66_test_x.npy', arr=xy_test[0][0])
np.save('../data/image/brain/npy/keras66_test_y.npy', arr=xy_test[0][1])

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
