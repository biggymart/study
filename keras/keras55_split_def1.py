import numpy as np
# split이 중요, 시계열은 y 데이터를 직접 잡아줘야 한다
# 태양광발전량 예측 dacon.io

#문제 1. 교재 p.221 참고 (다입력, 다:다)
dataset = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)): # 10
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy(dataset, 3, 1)
print(x, "\n", y)
# [[[ 1 11 21]
#   [ 2 12 22]
#   [ 3 13 23]]
#  [[ 2 12 22]
#   [ 3 13 23]
#   [ 4 14 24]]
#  [[ 3 13 23]
#   [ 4 14 24]
#   [ 5 15 25]]
#  [[ 4 14 24]
#   [ 5 15 25]
#   [ 6 16 26]]
#  [[ 5 15 25]
#   [ 6 16 26]
#   [ 7 17 27]]
#  [[ 6 16 26]
#   [ 7 17 27]
#   [ 8 18 28]]
#  [[ 7 17 27]
#   [ 8 18 28]
#   [ 9 19 29]]]
 
# [[[ 4 14 24]]
#  [[ 5 15 25]]
#  [[ 6 16 26]]
#  [[ 7 17 27]]
#  [[ 8 18 28]]
#  [[ 9 19 29]]
#  [[10 20 30]]]
print(x.shape) # (7, 3, 3)
print(y.shape) # (7, 1, 3)


# 이하 교재 코드 정리
# p. 207 (다:1)
# RNN의 input_shape = (samples, time_steps, features)
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10, )

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
            break
        tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy1(dataset, 4)
print(x, "\n", y)


# p. 211 (다:다)
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # (10, )

def split_xy2(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

time_steps = 4
y_column = 2
x, y = split_xy2(dataset, time_steps, y_column)
print(x, "\n", y)
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]]

# [[ 5  6]
#  [ 6  7]
#  [ 7  8]
#  [ 8  9]
#  [ 9 10]]
print("x.shape :", x.shape) # (5, 4)
print("y.shape :", y.shape) # (5, 2)


# p. 214, 219 (다입력, 다:1)
dataset = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column - 1

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy3(dataset, 3, 1) # (dataset, 3, 2) -> y 행 2줄 사용
print(x, "\n", y)
# [[[ 1 11]
#   [ 2 12]
#   [ 3 13]]
#  [[ 2 12]
#   [ 3 13]
#   [ 4 14]]
#  [[ 3 13]
#   [ 4 14]
#   [ 5 15]]
#  [[ 4 14]
#   [ 5 15]
#   [ 6 16]]
#  [[ 5 15]
#   [ 6 16]
#   [ 7 17]]
#  [[ 6 16]
#   [ 7 17]
#   [ 8 18]]
#  [[ 7 17]
#   [ 8 18]
#   [ 9 19]]
#  [[ 8 18]
#   [ 9 19]
#   [10 20]]]

#  [[23]
#  [24]
#  [25]
#  [26]
#  [27]
#  [28]
#  [29]
#  [30]]
print(x.shape) # (8, 3, 2)
print(y.shape) # (8, 1)
y = y.reshape(y.shape[0]) # y 값은 x의 샘플 수에 각 대응을 해야 함
print(y.shape) # (8,)

