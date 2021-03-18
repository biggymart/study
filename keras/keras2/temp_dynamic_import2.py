import importlib
from tensorflow.keras import applications as ap

# DIR
mother_mod = 'tensorflow.keras.applications.'

tmp = dir(ap)[:26]
MODULE_NAME = tmp
# print(MODULE_NAME) 
# ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception']
# '''

count = 0
for i in MODULE_NAME:
    model = importlib.import_module(mother_mod + i())
    model.trainable = True
    count += 1

    print(count, i)
    print(len(model.weights))
    print(len(model.trainable_weights))



# THE stackoverflow that solved my problem
# https://stackoverflow.com/questions/24940545/import-modules-from-a-list-in-python/54284181
# looping through list of functions in a function in python dynamically
# https://stackoverflow.com/questions/39422641/looping-through-list-of-functions-in-a-function-in-python-dynamically
# '''

from inspect import getmembers, isfunction

from somemodule import foo
print(getmembers(foo, isfunction))