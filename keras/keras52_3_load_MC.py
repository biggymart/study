# PREREQUISITE: change the hdf5 file's name as below (refer to checkpoint section)
# keras52_2_load_weight.py 카피

# ====== This part is shared =====================================
#1. data                                                          #
import numpy as np                                                #
from tensorflow.keras.datasets import mnist                       #
(x_train, y_train), (x_test, y_test) = mnist.load_data()          #
                                                                  #
#1-1. preprocessing                                               #
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.#
x_test = x_test.reshape(10000, 28, 28, 1)/255.                    #
                                                                  #
from tensorflow.keras.utils import to_categorical                 #
y_train = to_categorical(y_train)                                 #
y_test = to_categorical(y_test)                                   #
# ===== This part is shared ======================================

#2. model
#3. compile and fit
from tensorflow.keras.models import load_model
model_checkpoint = load_model('../data/modelCheckpoint/k52_1_mnist_checkpoint.hdf5') 
# The name of hdf5 file has been modified before running this file
# load both the model and the checkpoint

#4 evaluate and predict
result = model_checkpoint.evaluate(x_test, y_test)
print("MODEL_CHECKPOINT: [categorical_crossentropy, acc] :", result)

# Result of keras52_3_load_MC.py
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0764 - acc: 0.9762
# MODEL_CHECKPOINT: [categorical_crossentropy, acc] : [0.07644817233085632, 0.9761999845504761]

# model_checkpoint의 acc가 keras52_2_load_weight.py에서 본 model_saved보다 더 좋음
# 왜냐하면 model_saved는 early stopping의 patience만큼 지난 시점이지만
# checkpoint는 그만큼 더 앞선 시점에서 결정되기 때문이다