#1. data
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
scaler = MaxAbsScaler()
# scaler = PowerTransformer(method='yeo-johnson')
# scaler = PowerTransformer(method='box-cox')
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.max(x), np.min(x))
print(np.max(x[0]))


#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=150, verbose=1, validation_split=0.2)

#4. evalutate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=4)
print("mse :", mse, "\nmae :", mae)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# MinMaxScaler
# mse : 7.581524848937988 
# mae : 2.1726179122924805
# RMSE : 2.7534571507275873
# R2 : 0.9232851710250565

# StandardScaler
# mse : 8.463397979736328
# mae : 2.2769787311553955
# RMSE : 2.909192101656423
# R2 : 0.9143618107277143

# RobustScaler
# mse : 9.573953628540039
# mae : 2.4046943187713623
# RMSE : 3.0941808568991114
# R2 : 0.9031244756441291

# QuantileTransformer
# mse : 9.316068649291992
# mae : 2.180145502090454
# RMSE : 3.0522235266368343
# R2 : 0.9057339417045002

# MaxAbsScaler
# mse : 11.227215766906738
# mae : 2.692628860473633
# RMSE : 3.3507037671555873
# R2 : 0.8863957067373539

# PowerTransformer(method='yeo-johnson')
# mse : 7.429223537445068
# mae : 2.107837677001953
# RMSE : 2.7256604658844763
# R2 : 0.9248262549940945

# PowerTransformer(method='box-cox')
# ValueError: The Box-Cox transformation can only be applied to strictly positive data


# PowerTransformer
# 목표: 어떤 분포로부터 가우시안 분포에 근사하려고 함 (분산의 안정화 및 비대칭도의 최소화)
# 특징: 비선형, 각 값의 순서는 보존됨, parametric

# cf> Nonparametric
# Non-parametric models differ from parametric models in that the model structure is not specified a priori but is instead determined from data.
# Don't assume anything about the distribution.
# e.g> histogram, distribution free, median

# 가우시안 분포 == 종모양의 분포

# MaxAbsScaler
# 절댓값을 기준으로 최대인 값이 1.0이 되도록 함
# 데이터의 중앙값 변화 없음, 희소성 변화 없음