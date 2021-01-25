import pandas as pd
import numpy as np

#0. Pinball loss
import tensorflow.keras.backend as K
def quantile_loss_dacon(q, y_true, y_pred):
	err = (y_true - y_pred)
	return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#1. data
train = pd.read_csv('C:/data/csv/dacon/train/train.csv') # (52560, 9) == (24 * 2 * 1094, 9) 1094 days
submission = pd.read_csv('C:/data/csv/dacon/sample_submission.csv') # (7776, 10) == (24 * 2 * 2 * 81, 10) 81 sets of 2 days

def preprocess_data(data, is_train=True): # making Target columns 
    temp = data.copy()
    temp.insert(1,'Hour_Minute', data['Hour'] * 2 + data['Minute'] // 30)
    temp = temp[['Hour_Minute', 'TARGET', 'DHI', 'DNI', 'RH', 'T']] # 'WS' 상관계수 낮아서 제거

    if is_train==True: # train file here
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # new column for predicting tomorrow
        temp['Target2'] = temp['TARGET'].shift(-48 * 2).fillna(method='ffill') # for the day after tomorrow
        temp = temp.dropna()
        return temp.iloc[:-48 * 2] # excluding rows w/o columns Target1 and Target2 

    elif is_train==False: # test file here
        temp = temp[['Hour_Minute', 'TARGET', 'DHI', 'DNI', 'RH', 'T']] # 'WS' 상관계수 낮아서 제거
        return temp.iloc[-48:, :] # slice only Day6

# train data
df_train = preprocess_data(train) # after preprocessing: (52464, 9), with Target columns, 2 days at the end sliced
from sklearn.model_selection import train_test_split
x_train, x_eval, y1_train, y1_eval, y2_train, y2_eval = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], df_train.iloc[:, -1], test_size=0.2, random_state=0) # Target 1, (41971, 7), (10493, 7)

# test data
df_test = []
for i in range(81):
    file_path = 'C:/data/csv/dacon/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) # (48, 7) * 81 files
    df_test.append(temp)
x_test = pd.concat(df_test) # (3888, 7) == (48 * 81, 7)

x_train = x_train.to_numpy()
x_eval = x_eval.to_numpy()
x_test = x_test.to_numpy()

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_eval = scaler.transform(x_eval)
x_test = scaler.transform(x_test)

# reshape as RNN data
x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_eval = x_eval.reshape(-1, x_eval.shape[1], 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1)

#2. model
#3. compile and fit
#4. evaluate and predict
def RNN(x_train, x_eval, y_train, y_eval, x_test):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(x_train.shape[1], 1)))
    nodes = [32, 16, 8, 1]
    for i in nodes:
        model.add(Dense(i, activation='relu'))
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=6, mode='auto')
    rlr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    pred_return = pd.DataFrame() # [3888 rows x 9 columns], <class 'pandas.core.frame.DataFrame'>
    for i in range(len(quantiles)):
        from tensorflow.keras.optimizers import Adam
        model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(quantiles[i], y_true, y_pred), optimizer=Adam(learning_rate=0.001), metrics=['mae'])
        model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es, rlr], validation_data=(x_eval, y_eval))
        pred = model.predict(x_test) # <class 'numpy.ndarray'>, (3888, 1)
        pred_rs = pred.reshape(pred.shape[0]) # (3888, )
        pred_pd = pd.Series(pred_rs)
        pred_return = pd.concat([pred_return, pred_pd], axis=1)

    return pred_return, model

def train_data(x_train, x_eval, y_train, y_eval, x_test):
    RNN_models = []
    RNN_actual_pred = pd.DataFrame()
    pred, model = RNN(x_train, x_eval, y_train, y_eval, x_test)

    RNN_models.append(model)
    RNN_actual_pred = pd.concat([RNN_actual_pred, pred], axis=1)
    return RNN_models, RNN_actual_pred

# Flow Control
# Target1
models_1, results_1 = train_data(x_train, x_eval, y1_train, y1_eval, x_test) # <class 'pandas.core.frame.DataFrame'>, (3888, 1)

# Target2
models_2, results_2 = train_data(x_train, x_eval, y2_train, y2_eval, x_test) # <class 'pandas.core.frame.DataFrame'>, (3888, 1)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('C:/data/csv/dacon/submission_cyc.csv', index=False)

for model in models_1:
    loss1 = model.evaluate(x_eval, y1_eval)
    print(loss1)
for model in models_2:
    loss2 = model.evaluate(x_eval, y2_eval)
    print(loss2)

# [0.8175743222236633, 6.772909641265869]
# [0.7797744870185852, 6.870870113372803]

# Reference:
# https://towardsdatascience.com/deep-quantile-regression-c85481548b5a
# https://dacon.io/competitions/official/235680/codeshare/2300?page=1&dtype=recent&ptype=pub

'''
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True) # 사각형 형태, annotation 글씨 넣어주기, column bar
plt.show()
'''