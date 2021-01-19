import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)

def split_xy3(dataset2, x_row, x_col, y_row, y_col):
    x,y = list(), list()
    for i in range(len(dataset)):
        x_start_number = i
        x_end_number = i + x_row
        y_end_number = x_end_number + y_row -1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : x_col]
        tmp_y = dataset[x_end_number -1 : y_end_number, x_col :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset,3,2,4,1)
print(x, '\n', y)
print(x.shape)
print(y.shape)

'''
[[[ 1 11]
  [ 2 12]
  [ 3 13]]
 [[ 2 12]
  [ 3 13]
  [ 4 14]]
 [[ 3 13]
  [ 4 14]
  [ 5 15]]
 [[ 4 14]
  [ 5 15]
  [ 6 16]]
 [[ 5 15]
  [ 6 16]
  [ 7 17]]] 

[[[23]
  [24]
  [25]
  [26]]
 [[24]
  [25]
  [26]
  [27]]
 [[25]
  [26]
  [27]
  [28]]
 [[26]
  [27]
  [28]
  [29]]
 [[27]
  [28]
  [29]
  [30]]]
  
(5, 3, 2)
(5, 4, 1)
'''