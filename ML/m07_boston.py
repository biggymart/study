#1. data (회귀)
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

### 분기점 1 ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()  # Option 1
# scaler = StandardScaler() # Option 2
scaler.fit_transform(x_train)
scaler.transform(x_test)

### 분기점 2 ###
#2. model
from sklearn.neighbors import KNeighborsRegressor # Option 1
model = KNeighborsRegressor()
# from sklearn.tree import DecisionTreeRegressor # Option 2 
# model = DecisionTreeRegressor()
# from sklearn.ensemble import RandomForestRegressor # Option 3
# model = RandomForestRegressor()
# from sklearn.linear_model import LinearRegression # Option 4 (회귀)
# model = LinearRegression()

#3. fit
model.fit(x_train, y_train)

#4. score and predict
result = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score # 회귀
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)


### 결과 ###
# 1. MinMaxScaler
#   1-1. KNeighborsRegressor
#   r2_score : 0.5900872726222293
#   1-2. DecisionTreeRegressor
#   r2_score : 0.8058568011421567
#   1-3. RandomForestRegressor
#   r2_score : 0.9205336713604584
#   1-4. LinearRegression
#   r2_score : 0.8111288663608656
# 2. StandardScaler
#   2-1. KNeighborsRegressor
#   r2_score : 0.5900872726222293
#   2-2. DecisionTreeRegressor
#   r2_score : 0.7071911527557457
#   2-3. RandomForestRegressor
#   r2_score : 0.9190443873119692
#   2-4. LinearRegression
#   r2_score : 0.8111288663608656