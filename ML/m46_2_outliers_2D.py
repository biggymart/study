# 실습
# outliers1 함수는 1차원 데이터만 가능
# 행렬 형태도 적용 가능하도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
    [1000,2000,3,4000,5000,6000,7000,8,9000,10000]])

aaa = aaa.transpose()
# print(aaa.shape) # (10, 2)

def outliers(data_out):
    outliers_lst = []
    for i in range(data_out.shape[1]):
        print("========",i, "==========")
        data_col = data_out[:, i]
        quartile_1, q2, quartile_3 = np.percentile(data_col, [25, 50, 75]) 
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)

        outliers_lst.append(np.where((data_col > upper_bound) | (data_col < lower_bound)))

    return outliers_lst

print(outliers(aaa))


# LOF
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-outlier-detection-py

###### 시도했다가 삭제 #####
# 어떻게 하면 정규화를 할 수 있을까?
# http://hleecaster.com/ml-normalization-concept/

# 정규화 (Z-score)
# normalized_data_col = []
# for value in data_col:
#     normalized_value = (value - np.mean(data_col)) / np.std(data_col)
#     normalized_data_col.append(normalized_value)
# print(normalized_data_col)
############################