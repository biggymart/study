# 람다 함수의 정의

gradient = lambda x: 2*x - 4

def gradient2(x):
    temp = 2*x - 4
    return temp

x = 3 

print(gradient(x))
print(gradient2(x))


##################################

# 이차함수 시각화
import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100) # -1 부터 6까지 100개를 넣는다
y = f(x)

plt.plot(x, y, 'k-') # black line
plt.plot(2, 2, 'sk') # square black
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

##################################


import numpy as np

f = lambda x: x**2 - 4*x + 6
# gradient 는 f의 도함수
gradient = lambda x : 2*x - 4

# 랜덤한 x 값
x0 = 10.0
epoch = 30
learning_rate = 0.1

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0) 
    x0 = temp

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))

# lr을 줄이면 epoch를 늘려야 함