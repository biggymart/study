# 이미지 디렉토리: data/image/vgg/ 
# dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


#1. data
img_dog = load_img('../data/image/vgg/dog1.jpg', target_size=(224, 224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224, 224))
img_lion = load_img('../data/image/vgg/lion1.jpg', target_size=(224, 224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224, 224))
# plt.imshow(img_dog)
# plt.show()
# print(img_dog)
# <PIL.Image.Image image mode=RGB size=224x224 at 0x28B51F98070>

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
# print(arr_dog, arr_dog.shape)
# [[[150. 159. 166.] ...]] (224, 224, 3)
# print(type(arr_dog))
# <class 'numpy.ndarray'>

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
# print(arr_dog)
# [[[ 62.060997  42.221     26.32    ] ... ]]
# print(arr_dog.shape)
# (224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
# print(arr_input.shape)
# (4, 224, 224, 3)

#2. modeling
# 훈련시키려는 목적이 아니라 VGG16을 통해서 얼마나 잘 예측하는지 보려는 것
model = VGG16()
results = model.predict(arr_input)

# print(results)
# print('results.shape :', results.shape)
# results.shape : (4, 1000)
# 1000은 이미지넷에서 구분하는 카테고리의 개수

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions
decode_results = decode_predictions(results)
print('results[0] :', decode_results[0])
print('results[1] :', decode_results[1])
print('results[2] :', decode_results[2])
print('results[3] :', decode_results[3])

# results[0] : [('n02134084', 'ice_bear', 0.3722787), ('n02104029', 'kuvasz', 0.22540341), ('n02099712', 'Labrador_retriever', 0.17893106), ('n02099601', 'golden_retriever', 0.06741703), ('n02111500', 'Great_Pyrenees', 0.03272229)]
# results[1] : [('n02110185', 'Siberian_husky', 0.24078272), ('n02109961', 'Eskimo_dog', 0.19176963), ('n02091244', 'Ibizan_hound', 0.120515026), ('n02106030', 'collie', 0.07620275), ('n02087046', 'toy_terrier', 0.043094564)]
# results[2] : [('n02786058', 'Band_Aid', 0.12355975), ('n03291819', 'envelope', 0.08535222), ('n03314780', 'face_powder', 0.065920904), ('n04116512', 'rubber_eraser', 0.06510561), ('n04548280', 'wall_clock', 0.05533517)]
# results[3] : [('n04350905', 'suit', 0.94845945), ('n04591157', 'Windsor_tie', 0.0148617), ('n04479046', 'trench_coat', 0.014851726), ('n02883205', 'bow_tie', 0.005577891), ('n03594734', 'jean', 0.0030762057)]

# keras67 남자 여자 구별하는 것에 적용할 것


