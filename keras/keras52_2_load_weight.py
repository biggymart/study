# keras51_1_save_model.py 카피

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.activations import relu, softmax

model_now = Sequential()
model_now.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid', strides=1, input_shape=(28,28,1), activation=relu))
model_now.add(MaxPooling2D(pool_size=2, strides=(2,2)))
model_now.add(Flatten())
model_now.add(Dense(256, activation=relu))
model_now.add(Dropout(0.5))
model_now.add(Dense(10, activation=softmax))

#3. compile ONLY and load_weights
from tensorflow.keras.optimizers import Adam
model_now.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['acc']) 

model_now.load_weights('../data/h5/k52_1_weight.h5') # In a way, this line replaces fit

#4. evaluate
result1 = model_now.evaluate(x_test, y_test)
print("MODEL_NOW: [categorical_crossentropy, acc] :", result1)

# Result of model_now
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0695 - acc: 0.9772
# MODEL_NOW: [categorical_crossentropy, acc] : [0.06950126588344574, 0.9771999716758728]

##############################################
### Compare with what we have saved before ###
##############################################

from tensorflow.keras.models import load_model
model_saved = load_model('../data/h5/k52_1_model2.h5') # load both the model and the weight

#4 evaluate and predict
result2 = model_saved.evaluate(x_test, y_test)
print("MODEL_SAVED: [categorical_crossentropy, acc] :", result2)

# Results of keras52_2_load_weight.py
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0764 - acc: 0.9762
# MODEL_NOW: [categorical_crossentropy, acc] : [0.07644817233085632, 0.9761999845504761]
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0764 - acc: 0.9762
# MODEL_SAVED: [categorical_crossentropy, acc] : [0.07644817233085632, 0.9761999845504761]

# Compare with the result of keras52_1_save_weight.py
# 313/313 [==============================] - 0s 1ms/step - loss: 0.0764 - acc: 0.9762
# [categorical_crossentropy, acc] : [0.07644817233085632, 0.9761999845504761]
# ==> this is same as the above results

# LESSON LEARNED:
# (1) It is possible to save both the model and its weight after fit
# (2) Or, it is also possible to save only its weight after fit