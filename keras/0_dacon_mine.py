import pandas as pd
import numpy as np

#1. data
train = pd.read_csv('C:/data/csv/dacon/train/train.csv') # (52560, 9) == (24 * 2 * 1094, 9) 1094 days
submission = pd.read_csv('C:/data/csv/dacon/sample_submission.csv') # (7776, 10) == (24 * 2 * 2 * 81, 10) 81 sets of 2 days

def preprocess_data(data, is_train=True): # making Target columns 
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True: # train file here
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # new column for predicting tomorrow
        temp['Target2'] = temp['TARGET'].shift(-48 * 2).fillna(method='ffill') # for the day after tomorrow
        temp = temp.dropna()
        return temp.iloc[:-48 * 2] # excluding rows w/o columns Target1 and Target2 

    elif is_train==False: # test file here
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]     
        return temp.iloc[-48:, :] # slice only Day6

# load train data
df_train = preprocess_data(train) # after preprocessing (52464, 9), with Target columns, 2 days at the end sliced
from sklearn.model_selection import train_test_split
x1_train, x1_eval, y1_train, y1_eval = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.2, random_state=0) # Target 1, (41971, 7), (10493, 7)
x2_train, x2_eval, y2_train, y2_eval = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.2, random_state=0) # Target 2

# test data
df_test = []
for i in range(81):
    file_path = 'C:/data/csv/dacon/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False) # (48, 7) * 81 files
    df_test.append(temp)
x_test = pd.concat(df_test) # (3888, 7) == (48 * 81, 7)

# reshape as RNN data
x1_train = x1_train.to_numpy()
x1_train = x1_train.reshape(-1, x1_train.shape[1], 1)
x1_eval = x1_eval.to_numpy()
x1_eval = x1_eval.reshape(-1, x1_eval.shape[1], 1)
x2_train = x2_train.to_numpy()
x2_train = x2_train.reshape(-1, x2_train.shape[1], 1)
x2_eval = x2_eval.to_numpy()
x2_eval = x2_eval.reshape(-1, x2_eval.shape[1], 1)
x_test = x_test.to_numpy()
x_test = x_test.reshape(-1, x_test.shape[1], 1)

#2. model
#3. compile and fit
#4. evaluate and predict
def RNN(x_train, x_eval, y_train, y_eval, x_test):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
    model = Sequential()
    model.add(SimpleRNN(32, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.5))
    nodes = [16, 8, 4, 1]
    for i in nodes:
        model.add(Dense(i, activation='relu'))
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

    from tensorflow.keras.optimizers import Adam
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[es, rlr], validation_data=(x_eval, y_eval))
    pred = model.predict(x_test) # <class 'numpy.ndarray'>, (3888, 1)
    pred_rs = pred.reshape(pred.shape[0]) # (3888, )
    pred_return = pd.Series(pred_rs) # in order to transform ndarray into pd.Series, needs to be 1 dimension
    return pred_return, model

def train_data(x_train, x_eval, y_train, y_eval, x_test):
    RNN_models = []
    RNN_actual_pred = pd.DataFrame()
    pred, model = RNN(x_train, x_eval, y_train, y_eval, x_test)

    RNN_models.append(model)
    RNN_actual_pred = pd.concat([RNN_actual_pred, pred], axis=1)
    return RNN_models, RNN_actual_pred

# Target1
models_1, results_1 = train_data(x1_train, x1_eval, y1_train, y1_eval, x_test) 
# <class 'pandas.core.frame.DataFrame'>, (3888, 1)
results_1.sort_index()[:48]

# Target2
models_2, results_2 = train_data(x2_train, x2_eval, y2_train, y2_eval, x_test)
# <class 'pandas.core.frame.DataFrame'>, (3888, 1)
results_2.sort_index()[:48]
results_1.sort_index().iloc[:48]
results_2.sort_index()

print(results_1.shape, results_2.shape)
print(results_1, results_2)
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission.to_csv('C:/data/csv/dacon/submission_v3.csv', index=False)


# https://towardsdatascience.com/deep-quantile-regression-c85481548b5a
# https://dacon.io/competitions/official/235680/codeshare/2300?page=1&dtype=recent&ptype=pub