from tensorflow.keras import applications as ap
import importlib

# print(dir(ap))
# ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
#  'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'InceptionResNetV2', 'InceptionV3', 'MobileNet',
#  'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2',
#  'VGG16', 'VGG19', 'Xception', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__',
#  '__spec__', '_sys', 'densenet', 'efficientnet', 'imagenet_utils', 'inception_resnet_v2', 'inception_v3', 'mobilenet', 'mobilenet_v2',
#  'nasnet', 'resnet', 'resnet50', 'resnet_v2', 'vgg16', 'vgg19', 'xception']

# print(ap.DenseNet121())
# <tensorflow.python.keras.engine.functional.Functional object at 0x000001E87308C130>
# print(ap.densenet)
# <module 'tensorflow.keras.applications.densenet' from 'C:\\Users\\ai\\Anaconda3\\lib\\site-packages\\tensorflow\\keras\\applications\\densenet\\__init__.py'>
