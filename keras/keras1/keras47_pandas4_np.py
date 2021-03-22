import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)

print(df.shape) # (150, 5)
print(df.info()) # target만 int, 나머지는 float64

# pandas를 numpy로 바꾸는 것을 알아보아라
df_num1 = df.to_numpy() # copy=True 
print(type(df_num1)) # target 값이 float으로 바꼈지 (numpy는 하나의 형태만 용인함)

df_num2 = df.values
print(type(df_num2))

np.save('../data/npy/iris_sklearn.npy', arr=df_num1)

# 과제: pandas의 loc iloc 에 대해 정리하시오
