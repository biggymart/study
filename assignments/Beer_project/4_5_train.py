from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, DenseNet121, NASNetMobile
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

TRAIN_DIR = 'C:/data/image/beer'

PATCH_DIM = 224
images_num = 1250
batch_num = 8
steps_num = int(images_num / batch_num)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest', 
    validation_split=0.2  
)

xy_train = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(PATCH_DIM, PATCH_DIM),
    batch_size=batch_num, 
    class_mode='categorical',
    subset='training'
)

xy_val = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(PATCH_DIM, PATCH_DIM),
    batch_size=batch_num, 
    class_mode='categorical',
    subset='validation'
)

initial_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(PATCH_DIM, PATCH_DIM, 3)
)
last = initial_model.output

x = Flatten()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(4, activation='softmax')(x)

model = Model(initial_model.input, preds)

model.compile(loss='categorical_crossentropy', 
            optimizer='adam',
            metrics=['acc']
)

re = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2)
es = EarlyStopping(monitor='val_loss', patience=10)
hist = model.fit_generator(
    xy_train,
    steps_per_epoch= xy_train.samples // batch_num,
    epochs=100,
    validation_data= xy_val,
    validation_steps= xy_val.samples // batch_num,
    callbacks=[es, re]
)

model.save('C:/data/h5/train.h5')
acc = hist.history['acc']
loss = hist.history['loss']

print("acc :", acc[-1])
