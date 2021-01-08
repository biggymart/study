import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

#1. data
a = np.array(range(1,101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] 
        aaa.append(subset) # aaa.append([item for item in subset])와 같음
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset.shape) # (96, 5)

x = dataset[:, :4]
y = dataset[:, -1]
print(x.shape, y.shape) # (96, 4) (96,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (96, 4, 1)

#2. model
model = load_model('./model/save_keras35.h5') # shape=(4,1)
model.add(Dense(5, name='new_layer1'))
model.add(Dense(1, name='new_layer2'))

from tensorflow.keras.callbacks import EarlyStopping
es =EarlyStopping(monitor='loss', patience=10, mode='auto')

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['acc']) # 미친 척하고 acc 매트릭스 넣어보자
hist = model.fit(x, y, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])
# fit도 반환값이 있음

print(hist) # <tensorflow.python.keras.callbacks.History object at 0x0000020CA9066370>
print(hist.history.keys()) # dict_keys(['loss', 'val_loss']) # 매트릭스 넣고난 후 dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

print(hist.history['loss'])
# Epoch 56/1000 --> 56개 values 나옴
# [2230.297119140625, 2069.096923828125, 1839.35791015625, 1517.9593505859375, 1098.2816162109375, 682.154296875, 
# 522.0822143554688, 526.5164184570312, 357.71868896484375, 245.11178588867188, 227.0982208251953, 180.6772918701172, 
# 95.25787353515625, 35.4088020324707, 14.192497253417969, 3.762566328048706, 3.0235681533813477, 1.9110645055770874, 
# 1.084882140159607, 1.0181982517242432, 1.0936064720153809, 0.6226630806922913, 0.5387325286865234, 0.678337574005127, 
# 0.517957866191864, 0.42119237780570984, 0.6535382866859436, 0.39122825860977173, 0.423246830701828, 0.3988876938819885, 
# 0.331664115190506, 0.3275023400783539, 0.15487661957740784, 0.12528273463249207, 0.12683257460594177, 0.08665623515844345, 
# 0.06182171031832695, 0.07805967330932617, 0.05448200926184654, 0.02664082869887352, 0.01358459796756506, 0.011252821423113346, 
# 0.007817205972969532, 0.008177059702575207, 0.010087821632623672, 0.00767829455435276, 0.009176836349070072, 0.012046965770423412, 
# 0.009404520504176617, 0.010993599891662598, 0.01066691055893898, 0.018105218186974525, 0.018918827176094055, 0.016592154279351234, 
# 0.011043828912079334, 0.020854437723755836] 
# loss 값이 epoch 순서대로 나옴, key 넣으면 value 나옴

# graph
import matplotlib.pyplot as plt
plt.plot(hist.history['loss']) # train loss
plt.plot(hist.history['val_loss']) # val loss
plt.plot(hist.history['acc']) # train acc, 회귀모델이니까 0으로 바닥에 기는 게 정상
plt.plot(hist.history['val_acc']) # val acc
# 기본 문법: plt.plot(x, y)
# plt.plot(값 하나만) --> 이 경우 순서대로 찍어준 후 선으로 이어줌; 즉, plot() 안에 y 값이 들어간다고 할 수 있음

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 범례
plt.show()

# fit 하면서 검증하는 거라서 일반적으로 val_loss 가 loss보다 더 큼
# 과적합: loss와 val_loss가 너무 벌어졌는데, 다른 지표가 좋다면 잘못된 것 
'''
      loss  val_loss
경우1  0.09 0.9
경우2  0.9  0.91
'''
# 로스는 낮을수록 좋음
# 로스와 발로스를 비교해보면 (열 비교) 경우1이 둘 다 더 좋은데, 
# 각 경우에서 (행 비교) 로스와 발로스의 차이가 경우1은 차이가 크고 경우2는 차이가 적음
# 따라서 경우2가 더 좋음, 경우1은 과적합임 (검증할 때 개판임, "터진다")

# 하지만 통상적으로 귀찮으므로 발로스만 보고 비교하는 경우가 있음
# 경우1은 수정의 여지가 많은데 경우2는 수정의 여지가 많지 않음