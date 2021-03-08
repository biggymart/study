# 결측치 처리 (None 같은 거)
# 모델을 만들고 결측치에 대해 예측하여 채우는 전략
# 선형 회귀 및 시계열에 잘 됨

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021']
dates = pd.to_datetime(datastrs)
print(dates)
print('==============================')

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate() # 보간법
print(ts_intp_linear)
