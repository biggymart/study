import numpy as np
import pandas as pd

#0. data setting
features = ['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']
target_feature = '시가'
denomination_date = '2018-05-04'
# .iloc[-663:,:]
y_col = features.index(target_feature)
future_day_gap = 2

#1. basic functions
def load_df(dir):
    file = pd.read_csv(dir, index_col=0, header=0, engine='python', encoding='CP949', thousands=',')
    file[['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']]
    file = file.sort_index()
    return file

def join_dfs(df1, df2):
    df1 = df1.dropna()
    df2 = df2.dropna()

    joined_df = pd.concat([df1, df2], join="inner", verify_integrity=True)
    return joined_df

plus_one = ['등락률', '거래량', '금액(백만)', '신용비',	'개인',	'기관',	'외인(수량)', '외국계',	'프로그램',	'외인비']
def four_plus_one(df, feature):
    idx = plus_one.index(feature)

    df_fix_col = df.loc[:,'시가':'종가']
    df_sel_col = df.loc[:, plus_one[idx]]
    df_alltime = pd.concat([df_fix_col, df_sel_col], axis=1)
    df_after_deno = df_alltime.loc[denomination_date:,:]
    return df_after_deno

def four_plus_two(df, feature1, feature2):
    idx1 = plus_one.index(feature1)
    idx2 = plus_one.index(feature2)

    df_fix_col = df.loc[:,'시가':'종가']
    df_sel_col1 = df.loc[:, plus_one[idx1]]
    df_sel_col2 = df.loc[:, plus_one[idx2]]

    df_alltime = pd.concat([df_fix_col, df_sel_col1, df_sel_col2], axis=1)
    # df_after_deno = df_alltime.loc[denomination_date:,:] 위에 있는 함수 이 부분과 row 반드시 맞춰야 함
    df_after_deno = df_alltime.iloc[-663:,:] 
    return df_after_deno

def run_model(df1, df2):
    df1_numpy = df1.to_numpy()
    df2_numpy = df2.to_numpy()

    #1-0. slicing for prediction
    x1_npy = df1_numpy[:-1 * future_day_gap, :].astype(np.float64)
    x2_npy = df2_numpy[:-1 * future_day_gap, :].astype(np.float64)
    y1_npy = df1_numpy[1* future_day_gap:, y_col].astype(np.float64)


    #1-0. preprocessing (LSTM)
    from sklearn.model_selection import train_test_split
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1_npy, y1_npy, train_size=0.8, shuffle=True)
    x2_train, x2_test = train_test_split(x2_npy, train_size=0.8, shuffle=True)
    x1_pred = df1_numpy[-1 * future_day_gap:]
    x2_pred = df2_numpy[-1 * future_day_gap:]

    from sklearn.preprocessing import MinMaxScaler

    x1_list = [x1_train, x1_test, x1_pred]
    x2_list = [x2_train, x2_test, x2_pred]
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    # element at 0 index in data should be the scaler-setter (usually x_train)
    def minmax_for_RNN(data, scaler):
        scaler.fit(data[0])
        for i in range(len(data)):
            data[i] = scaler.transform(data[i])
            data[i] = data[i].reshape(-1, data[i].shape[1], 1)
    minmax_for_RNN(x1_list, scaler1)
    minmax_for_RNN(x2_list, scaler2)

    # print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y1_train.shape, y1_test.shape)
    #      (528, 5)         (133, 5)       (528, 6)        (133, 6)       (528,)          (133,)

    np.savez('../data/npy/samsung_inverse1.npz', x1_test=x1_test, y1_test=y1_test, x1_pred=x1_pred)
    np.savez('../data/npy/samsung_inverse2.npz', x2_test=x2_test, x2_pred=x2_pred)

    #2. model
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, concatenate

    input1 = Input(shape=(x1_npy.shape[1], 1))
    dense1 = LSTM(300, activation='relu')(input1)
    dense1 = Dense(250, activation='relu')(dense1)
    dense1 = Dense(150, activation='relu')(dense1)
    dense1 = Dense(150, activation='relu')(dense1)
    dense1 = Dense(30, activation='relu')(dense1)

    input2 = Input(shape=(x2_npy.shape[1], 1))
    dense2 = LSTM(300, activation='relu')(input2)
    dense2 = Dense(250, activation='relu')(dense2)
    dense2 = Dense(150, activation='relu')(dense2)
    dense2 = Dense(150, activation='relu')(dense2)
    dense2 = Dense(30, activation='relu')(dense2)

    merge1 = concatenate([dense1, dense2])
    output1 = Dense(32)(merge1)
    output1 = Dense(16)(output1)
    output1 = Dense(16)(output1)
    output1 = Dense(16)(output1)
    output1 = Dense(1)(output1)

    model = Model(inputs=[input1, input2], outputs=output1)

    #3. compile and fit
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    modelpath = '../data/modelCheckpoint/samsung_inverse_{epoch:02d}-{val_loss:.4f}.hdf5'
    cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='loss', patience=30, mode='auto')

    model.compile(loss='mse', optimizer='adam', metrics='mae')
    model.fit([x1_train, x2_train], y1_train, epochs=1000, batch_size=32, verbose=1, callbacks=[es, cp], validation_split=0.2)

    model.save('../data/h5/samsung_inverse.h5')

    # from tensorflow.keras.models import load_model
    # model = load_model('../data/h5/samsung_lstm.h5')

    #4. evaluate and predict
    loss = model.evaluate([x1_test, x2_test], y1_test, batch_size=32)
    y_pred = model.predict([x1_pred, x2_pred], batch_size=32)

    return(loss, y_pred)

#1. flow control
rdata_samsung1 = load_df('C:\data\csv\samsung.csv')
rdata_samsung2 = load_df('C:\data\csv\samsung2.csv')
data_samsung = join_dfs(rdata_samsung1, rdata_samsung2)
data_inverse = load_df('C:\data\csv\kosdaq_inverse.csv')
# 위에도 있지만 참고로, plus_one = ['등락률', '거래량', '금액(백만)', '신용비',	'개인',	'기관',	'외인(수량)', '외국계',	'프로그램',	'외인비']
df_sam_featured = four_plus_one(data_samsung, plus_one[1]) # 거래량
df_inv_featured = four_plus_two(data_inverse, plus_one[1], plus_one[2]) # 거래량, 금액(백만)

result = run_model(df_sam_featured, df_inv_featured)
print("loss(mse, mae)", result[0], "\npredict", result[1], sep='')

# 결과 trial=1
# loss(mse, mae) [2394344.25, 1193.716064453125] 
# predict [[94473.625] [93075.484]]

# 결과 trial=2
# loss(mse, mae)[2672107.0, 1321.3719482421875]
# predict[[94703.24] [93382.99]]

# 결과 trial=3
# loss(mse, mae)[3970264.75, 1537.30615234375]
# predict[[93362.77] [92040.91]]

# 결과 trial=4
# loss(mse, mae)[3929277.75, 1491.656005859375]
# predict[[91563.38] [90250.57]]

# 결과 trial=5
# loss(mse, mae)[2528967.25, 1164.7802734375]
# predict[[89505.] [87874.54]]