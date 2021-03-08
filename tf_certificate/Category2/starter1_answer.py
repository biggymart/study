# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPool1D
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping


    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    x_train = x_train/255.
    x_test = x_test/255.

    y_train = to_categorical(y_train)   #(60000, 28, 28) (60000,)
    y_test = to_categorical(y_test)     #(10000, 28, 28) (10000,)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, input_shape=(28,28), activation='relu', padding='same'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    er = EarlyStopping(monitor='val_loss', patience=15, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[er])

    loss = model.evaluate(x_test, y_test) 
    print('loss, acc : ', loss)
    # loss, acc :  [0.41283154487609863, 0.8877000212669373]

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./tf_certificate/Category2/mymodel.h5")
