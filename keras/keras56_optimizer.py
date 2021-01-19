import numpy as np

#1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. model
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. compile and fit
#4. evaluate and predict
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

rate = [0.1, 0.01, 0.001, 0.0001]
# Adam, Adadelta, Adamax, Adagrad, RMSprop, SGD, Nadam
for i in rate:
    optimizer = Adam(lr=i) # Default value of learning rate is 0.001

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    model.fit(x, y, epochs=100, batch_size=1, verbose=0)

    loss, mse = model.evaluate(x, y, batch_size=1)
    y_pred = model.predict([11])
    print("lr = {0}".format(i), "\nloss :", loss, "result :", y_pred, sep='')

# Optimizer는 Gradient descent(경사하강법)에 기반함

# Adam
# lr = 0.1
# loss :847.0953369140625result :[[-32.22505]]
# lr = 0.01
# loss :0.0002564963942859322result :[[11.023906]]
# lr = 0.001
# loss :0.1516667902469635result :[[10.697695]]
# lr = 0.0001
# loss :187.90045166015625result :[[-10.283033]]

# Adadelta
# lr = 0.1
# loss :0.00033150558010675013result :[[11.032441]]
# lr = 0.01
# loss :5.312898792908527e-06result :[[11.003597]]
# lr = 0.001
# loss :9.49995637711254e-12result :[[10.999995]]
# lr = 0.0001
# loss :2.5423218218134647e-12result :[[11.]]

# Adamax
# lr = 0.1
# loss :7.797320365905762result :[[14.041817]]
# lr = 0.01
# loss :3.299479089946544e-07result :[[11.000008]]
# lr = 0.001
# loss :1.5523881069512413e-09result :[[10.999968]]
# lr = 0.0001
# loss :7.03113678390821e-10result :[[10.999988]]

# Adagrad
# lr = 0.1
# loss :29.78278160095215result :[[17.640085]]
# lr = 0.01
# loss :0.0018421746790409088result :[[11.070507]]
# lr = 0.001
# loss :2.4786757091277423e-09result :[[10.999966]]
# lr = 0.0001
# loss :4.2891343254858327e-10result :[[11.000028]]

# RMSprop
# lr = 0.1
# loss :4390055.0result :[[-3726.3738]]
# lr = 0.01
# loss :5210.4599609375result :[[-65.33367]]
# lr = 0.001
# loss :1.4578651189804077result :[[8.596521]]
# lr = 0.0001
# loss :0.013627193868160248result :[[11.123294]]

# SGD
# lr = 0.1
# loss :nanresult :[[nan]]
# lr = 0.01
# loss :nanresult :[[nan]]
# lr = 0.001
# loss :nanresult :[[nan]]
# lr = 0.0001
# loss :nanresult :[[nan]]

# Nadam
# lr = 0.1
# loss :6.642536845902214e-06result :[[10.994555]]
# lr = 0.01
# loss :3.2095886126626283e-07result :[[10.99907]]
# lr = 0.001
# loss :9.50158174362059e-09result :[[11.000119]]
# lr = 0.0001
# loss :9.957491329259938e-07result :[[11.001843]]