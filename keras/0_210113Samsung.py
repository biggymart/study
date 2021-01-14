#1. data
#1-pre. extract and refine data from csv
import numpy as np
import pandas as pd
df = pd.read_csv('../data/csv/samsung.csv', index_col=0, header=0, engine='python', encoding='CP949', thousands=',')
df_ordered = df[::-1]

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
x_pandas, y_pandas = mk_data('등락률') # 여기 반복문 여지 있음

x_npy = x_pandas.to_numpy() # (661, 5) <class 'numpy.ndarray'>
y_npy = y_pandas.to_numpy() # (661,) <class 'numpy.ndarray'>

x_npy = x_npy.astype(np.float64)
y_npy = y_npy.astype(np.float64)

#1-0. preprocessing (LSTM)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_npy, y_npy, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

np.save('../data/npy/samsung_x_test.npy', arr=x_test)
np.save('../data/npy/samsung_y_test.npy', arr=y_test)

# x_test = np.load('../data/npy/samsung_x_test.npy')
# y_test = np.load('../data/npy/samsung_y_test.npy')

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(400, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Dense(250, activation='relu'))
model.add(Dense(150, activation='relu')) 
model.add(Dense(150, activation='relu')) 
model.add(Dense(30, activation='relu')) 
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/samsung_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[es, cp], validation_split=0.2)

model.save('../data/h5/samsung_lstm.h5')

# from tensorflow.keras.models import load_model
# model = load_model('../data/h5/samsung_lstm.h5')

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=32)
print("loss(mse, mae) :", loss)

x_pred = np.array([89800, 91200, 89100, 89700, -0.99])
x_pred = x_pred.reshape(1, -1)
x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

np.save('../data/npy/samsung_x_pred.npy', arr=x_pred)
# x_pred = np.load('../data/npy/samsung_x_pred.npy')

y_pred = model.predict(x_pred, batch_size=32)
print(y_pred)

# 결과
# loss(mse, mae) : [1326206.25, 860.6065063476562]
# [[93747.92]]