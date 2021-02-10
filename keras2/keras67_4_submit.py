# 이 사람은 남자인가, 여자인가?
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model

try:
    DIMENSION = 256
    CHANNEL = 3
    user_path = input("Enter the image file directory :")
    X_input = np.empty((1, DIMENSION, DIMENSION, CHANNEL))
    img = image.load_img(user_path, target_size=(DIMENSION, DIMENSION))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    X_input[0,:,:,:] = x

    # predict
    model = load_model('../data/h5/k67_1_male_female_good.h5')

    y_pred = model.predict(X_input)
    prob = float(y_pred)

    if prob >= 0.5:
        print("이 사람이 남자일 확률은 :", prob * 100, "% 입니다.")
    else:
        print("이 사람이 여자일 확률은 :", (1-prob) * 100, "% 입니다.")

except FileNotFoundError:
    print("파일을 찾지 못했습니다. 정확한 디렉토리를 입력해주세요.")