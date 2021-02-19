import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# data from selenium


# load model (teachable machine)
# model_name = 'keras_model_beer'
# model = load_model('../data/h5/{0}.h5'.format(model_name))

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')


# predict and filter out
test_img_dir = 'C:/data/image/beer_selenium/cass/'
file_model_name = 'img_14.jpg'


img_path = test_img_dir + file_model_name
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])