# 시계열을 자르기 좋은 함수
import numpy as np

a = np.array(range(1,11)) # [ 1  2  3  4  5  6  7  8  9 10]
size = 5 # subset의 크기 (요소 갯수)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] 
        aaa.append(subset) # aaa.append([item for item in subset])와 같음
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("===================")
print(dataset)