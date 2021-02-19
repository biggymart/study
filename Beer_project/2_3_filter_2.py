# SIFT 필터로만 잘 안 걸러지는 듯 해서
# sigmoid로 구성된 모델로 한 번 걸러보도록 하자.


# load model (teachable machine)
# model_name = 'keras_model_beer'
# model = load_model('../data/h5/{0}.h5'.format(model_name))

from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
model = mobilenet_v2(weights='imagenet')


# predict and filter out



img_path = test_img_dir + file_model_name
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])






# import libraries
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

image_path = 'test1.JPG' # 여기에는 테스트할 이미지의 경로 및 이름을 넣어주시면 됩니다. 
im = cv2.imread(image_path) # 이미지 읽기


# object detection (물체 검출)
bbox, label, conf = cv.detect_common_objects(im)

print(bbox, label, conf)

im = draw_bbox(im, bbox, label, conf) 


cv2.imwrite('result.jpg', im) # 이미지 쓰기





