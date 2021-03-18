from tensorflow.keras.applications import *
# 전이학습 모델 전부
# https://keras.io/api/applications/
# 전이학습에 대한 튜토리얼
# https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751

model = EfficientNetB7()
model.trainable = True

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

### trainable = True
# Xception  VGG16     VGG19
# 236       32        38
# 156       32        38

# ResNet50  ResNet101   ResNet152   ResNet50V2  ResNet101V2 ResNet152V2
# 320       626         932         272         544         816
# 214       418         622         174         344         514

# InceptionV3   InceptionResNetV2   MobileNet   MobileNetV2 
# 378           898                 137         262         
# 190           490                 83          158         

# DenseNet121   DenseNet169 DenseNet201 
# 606           846         1006        
# 364           508         604         

# NASNetLarge   NASNetMobile
# 1546          1126
# 1018          742

# EfficientNetB0    B1  B2  B3  B4  B5  B6  B7
# 314               442 442 499 613 741 855 1040
# 213               301 301 340 418 506 584 711
