#1. data
#1-pre. extract and refine data from csv
import numpy as np
import pandas as pd
df = pd.read_csv('../data/csv/samsung.csv', index_col=0, header=0, engine='python', encoding='CP949')
df_ordered = df[::-1]

from pandas.api.types import is_string_dtype
lst = list(df_ordered.columns)
for i in range(len(lst)): # remove comma in string data
    if(is_string_dtype(df_ordered[lst[i]])):
        df_ordered[lst[i]] = df_ordered[lst[i]].str.replace(',','')

plus_one = ['등락률', '거래량', '금액(백만)', '신용비',	'개인',	'기관',	'외인(수량)', '외국계',	'프로그램',	'외인비']
def mk_data(pick):
    idx = plus_one.index(pick)

    # 시가, 고가, 저가, 종가, 거래량, 금액(백만) & 액면분할 이후
    df_fix_col = df_ordered.loc[:,'시가':'종가']
    df_sel_col = df_ordered.loc[:, plus_one[idx]]
    df_part1 = df_fix_col.iloc[1738:] # For some unknown reason, .loc['20180504'] does not function as expected
    df_part2 = df_sel_col.iloc[1738:]

    my_df = pd.concat([df_part1, df_part2], axis=1) # 좌우로 연결

    x_df = my_df.iloc[:-1] # (661, 5); <class 'pandas.core.frame.DataFrame'>
    y_pre = my_df.loc[:, '종가']
    y_se = y_pre.iloc[1:] # (661,); <class 'pandas.core.series.Series'>

    return x_df, y_se

# 이제 4개의 고정 features + 1개의 선택 feature의 x, y 데이터 구성 가능
x_pandas, y_pandas = mk_data('거래량')
'''
# 여기 반복문 여지 있음
'''
x_npy = x_pandas.to_numpy() # (661, 5) <class 'numpy.ndarray'>
y_npy = y_pandas.to_numpy() # (661,) <class 'numpy.ndarray'>

# x_npy = x_npy_str.astype(np.float)
# y_npy = y_npy_str.astype(np.float)
# ===== DATA READY ===== #

#1-0. preprocessing (LSTM)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(5, 1)))
model.add(Dropout(0.2))
model.add(Dense(20)) 
model.add(Dense(10)) 
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=100, callbacks=[es, cp], validation_split=0.2)

# model.save('../data/h5/samsung_lstm.h5')

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=100)
print("loss(mse, mae) :", loss)

x_pred = np.array[89800, 91200, 89100, 89700, 34161101]
scaler.transform(x_pred)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
y_pred = model.predict(x_pred, batch_size=100)
print(y_pred)

# Function call stack:
# train_function