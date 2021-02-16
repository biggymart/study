# pip install selenium
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
from keras import Sequential

def google_crawling(find_namelist):
    for i in range(2):
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_elements_by_css_selector(".mye4qd").click()
                except:
                    break
            last_height = new_height
        images = driver.find_element_by_css_selector(".rg_i_Q4LuWd")
        count = 1
        for image in images:
            try:
                image.click()
                time.sleep(1)
                imgUrl = driver.find_element_by_xpath('xpath').get_attribute("src")
                urllib.request.urlretrieve(imgUrl, path + str(count) + ".jpg")
                count += 1

xml_path1 =  "xpath"
face_classifier = cv2.CascadeClassifier(xml_path1)
def face_extractor(img):
    gray = cv2.cvtColor(img.cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is():
        return img
    crop_face = []
    point_x = []
    point_y = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
        crop_face.append(roi)
        point_x.append(x)
        point_y.append(y)
        # cropped_face = img[y:y+h, x:x+w]
        return img

count = 1
image_path1 = "./Common_data/image_naver/1/"
for i in range(1000):
    img = cv2.imread(image_path1 + str(i + 1) + ".jpg")
    if face_extractor(img) is not None:
        face = cv2.resize(face_extractor(img), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        file_name_path = './Common_data/image_all/1/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        count += 1

train_datagen = ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    brightness_range = [0.1, 1.0],
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

def build_model(optinizer='adam', lr=0.001):
    inputs = Input(shape=(200,200,3), name='input')
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(4, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    optimizers = [Adam, Adadelta, RMSprop, Adamax, Adagrad, SGD, Nadam]
    lr=[0.1, 0.01, 0.001]
    return {"optimizer":optimizers, "lr":lr}

'''
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(200,200,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


inceptionResNetV2 = inceptionResNetV2(
    include_top=False,
    input_shape=(xy_train[0][0].shape[1], xy_train[0][0].shape[2], xy_train[0][0].shape[3])
)
inceptionResNetV2.trainable = False
model = Sequential()
model.add(inceptionResNetV2)
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''

early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=5,
    factor=0.5,
    verbose=1
)
check_point = ModelCheckpoint(
    filepath = modelpath,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)
hist = model.fit_generator(
    xy_train,
    steps_per_epoch=len(xy_train),
    validation_data=(xy_test),
    validation_steps=len(xy_test),
    epochs=150,
    verbose=1,
    callbacks=[reduce_lr, EarlyStopping, check_point]
)
try:
    result = model.predict(face)
    result = np.argmax(result, axis=1)
    bss1 = []
    for i in range(len(result)):
        if result[i] == 0:
            bss1.append([cv2.putText(image, "Bae", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])
        elif result[i] == 1:
            bss1.append([cv2.putText(image, "Nam", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])
        elif result[i] == 2:
            bss1.append([cv2.putText(image, "Ha", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])
        elif result[i] == 3:
            bss1.append([cv2.putText(image, "Kang", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])
    bss1[:]
    cv2.imshow('Face Cropper', image)
except:
    cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Face Cropper', image)
    pass
if cv2.waitKey(1)== 13:
    break
movie.release()
cv2.destroyAllWindows()

# dlib.shape_predictor
# face_recognition_model_v1
# recognition