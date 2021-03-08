# 이상치 처리 (위치를 활용)
# 1. 0으로 처리 
# 2. NaN으로 바꾼 뒤 보간

# 결측치 및 이상치 처리 관련 블로그
# https://blog.naver.com/jju1213/222112651208


# 5000, 10000을 이상치로 간주하게끔 만든 임의적 데이터
import numpy as np
aaa = np.array([1,2,3,4,6,7,90,100,250,350])


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75]) 
    # q2 == 중위값; 25,50,75 프로 지점 반환해줘
    print("1사분위 :", quartile_1)
    print("q2 :", q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1 # InterQuartileRange
    print("iqr :", iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # 1사분위 아래로 iqr 1.5배까지는 정상이라 봐줄게
    upper_bound = quartile_3 + (iqr * 1.5) # 3사분위 위로 iqr 1.5배까지는 정상이라 봐줄게
    return np.where((data_out > upper_bound) | (data_out < lower_bound)) # 인덱스 반환

outlier_loc = outliers(aaa)
print("이상치의 위치 :", outlier_loc)
# 이상치의 위치 : (array([8, 9], dtype=int64),)
# "인덱스 8, 9에 위치한 것은 이상치야"


# quantile, quartile에 대한 블로그
# https://blog.naver.com/tonyhuh/222166249684
# quantile는 데이터 셋은 같은 사이즈의 데이터로 나줘 줌

from scipy.stats import iqr
interquartile_range = iqr(aaa)
print("stats library 활용한 iqr :", interquartile_range)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
