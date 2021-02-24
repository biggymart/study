import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.models import load_model

DIMENSION = 224
CHANNEL = 3
user_path = 'C:/Users/ai/Desktop/pred.jpg'
X_input = np.empty((1, DIMENSION, DIMENSION, CHANNEL))
img = image.load_img(user_path, target_size=(DIMENSION, DIMENSION))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
x = np.squeeze(x)
X_input[0,:,:,:] = x

# predict
model = load_model('../data/h5/beer_NASNetMobile_epoch77_acc94.h5')
y_pred = model.predict(X_input)
class_names = ['cass', 'filgood', 'filite', 'hite']

print("This looks like", class_names[np.argmax(y_pred)])
