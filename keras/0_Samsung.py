#1. data
#1-pre. extract and refine data from csv
import numpy as np
import pandas as pd
df_old = pd.read_csv('../data/csv/samsung.csv', index_col=0, header=0, engine='python', encoding='CP949', thousands=',')
df_new = pd.read_csv('../data/csv/samsung2.csv', index_col=0, header=0, engine='python', encoding='CP949', thousands=',', nrows=1)
df = pd.concat([df_old[::-1], df_new], join="inner")

plus_one = ['등락률', '거래량', '금액(백만)', '신용비',	'개인',	'기관',	'외인(수량)', '외국계',	'프로그램',	'외인비']
def mk_data(pick): # features: 4 plus 1
    idx = plus_one.index(pick)

    df_fix_col = df.loc[:,'시가':'종가']
    df_sel_col = df.loc[:, plus_one[idx]]
    df_alltime = pd.concat([df_fix_col, df_sel_col], axis=1)
    df_after_deno = df_alltime.iloc[1738:] # For some unknown reason, .loc['20180504'] does not function as expected

    return df_after_deno

# 이제 4개의 고정 features + 1개의 선택 feature의 x, y 데이터 구성 가능
for count, feature in enumerate(plus_one):
    df_pandas = mk_data(feature).to_numpy()

    x_npy = df_pandas[:-1, :].astype(np.float64) # (662, 5)
    y_npy = df_pandas[1:, 3].astype(np.float64)  # (662, )

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

    x_pred = df_pandas[-1]
    x_pred = x_pred.reshape(1, -1)
    x_pred = scaler.transform(x_pred)
    x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
    
    np.savez('../data/npy/samsung_{0}.npz'.format(count), x_test=x_test, y_test=y_test, x_pred=x_pred)
    # npz_loaded = np.load('../data/npy/samsung.npz')
    # x_test = npz_loaded['x_test']
    # y_test = npz_loaded['y_test']
    # x_pred = npz_loaded['x_pred']

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
    model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=0, callbacks=[es, cp], validation_split=0.2)

    model.save('../data/h5/samsung_lstm_{0}.h5'.format(count))

    # from tensorflow.keras.models import load_model
    # model = load_model('../data/h5/samsung_lstm.h5')

    #4. evaluate and predict
    loss = model.evaluate(x_test, y_test, batch_size=32)
    print(count, "loss(mse, mae) :", loss)

    y_pred = model.predict(x_pred, batch_size=32)
    print(count, y_pred)

# 결과 0113일자
# loss(mse, mae) : [1326206.25, 860.6065063476562]
# [[93747.92]]

# # 결과 0114일자
# 0 loss(mse, mae) : [1533577.625, 885.9054565429688]
# 0 [[92199.19]]
# 1 loss(mse, mae) : [2204966.75, 1173.6610107421875]
# 1 [[90640.2]]
# 2 loss(mse, mae) : [1518838.625, 923.9739379882812]
# 2 [[91774.89]]
# 3 loss(mse, mae) : [1546232.0, 928.6940307617188]
# 3 [[91110.99]]
# 4 loss(mse, mae) : [1348317.25, 879.5555419921875]
# 4 [[91197.48]]
# 5 loss(mse, mae) : [1579582.125, 998.9874267578125]
# 5 [[92360.664]]
# 6 loss(mse, mae) : [1238977.375, 834.915771484375]
# 6 [[92103.77]]
# 7 loss(mse, mae) : [1519749.25, 939.0438232421875]
# 7 [[91279.58]]
# 8 loss(mse, mae) : [1947028.25, 1055.9395751953125]
# 8 [[90833.93]]
# 9 loss(mse, mae) : [1575937.125, 931.3853759765625]
# 9 [[91676.01]]