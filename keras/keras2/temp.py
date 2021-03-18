from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# conv_base = VGG16(include_top=False)
# conv_base.trainable = False

# for i, layer in enumerate(conv_base.layers):
#        print(i, layer.name)

# for layer in model.layers[:249]:
#        layer.trainable = False
# for layer in model.layers[249:]:      
#    layer.trainable = True

# set_trainable = False
# for layer in conv_base.layers:
#     print(layer.name)
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

# conv_base.summary()