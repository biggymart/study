import numpy as np
# def split_xy(Data, x_col, x_row, y_col, y_row)

#문제2.
dataset = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)
print(dataset)
print(dataset.shape)
print(dataset[0])

# def split_xy1(dataset, x_row, x_col, y_row, y_col):
#     x_lst, y_lst = list(), list()
#     for i in range(len(dataset)):
#         x_row_end = i + x_row
#         x_col_end = i + x_col
#         y_row_end = i + y_row
#         y_col_end = i + y_col
#         # if y_end_number > len(dataset):
#             # break
#         tmp_x = dataset[i:x_row_end, i:x_col_end]
#         tmp_y = dataset[i:y_row_end, i:y_col_end]
#         x_lst.append(tmp_x)
#         y_lst.append(tmp_y)
#     return np.array(x_lst), np.array(y_lst)
# x, y = split_xy1(dataset, 2, 2, 2 ,2)
# print(x)
# print(y)


#   return