# 활성화 함수: 다음 레이어로 넘겨주는 역할
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

# 레이어의 마지막 값에 이렇게 곱해줌 (결과값은 0~1 사이)
# compile해서 binary crossentropy를 설정해줘야 0.5 기준으로 판별해줌

x = np.arange(-5, 5, 0.1) # -5, 5 사이; 0.1 간격으로
y = sigmoid(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()


#####################################

def tanh(x):
    return (np.exp(x) - np.exp(-x)/(np.exp(x) + np.exp(-x)))

x = np.arange(-5, 5, 0.1)
y = tanh(x) # LSTM 내부에 들어가 있었음

plt.plot(x, y)
plt.grid()
plt.show()

#####################################

def relu(x):
    return np.maximum(0, x)

def Leaky_ReLU(x):
    return np.maximum(0.01 * x, x)

def elu(x, alp):
    return (x>0) * x + (x<=0) * (alp*(np.exp(x) - 1))

def swish(x):
    return x * tf.nn.sigmoid(x)

def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


x = np.arange(-5, 5, 0.1)
y = relu(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()

#####################################

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
# 전부합치면 1

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()