import pandas as pd
import numpy as np

#1. data
train = pd.read_csv('C:/data/csv/dacon/train/train.csv')
submission = pd.read_csv('C:/data/csv/dacon/sample_submission.csv')

def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill') # new column for predicting tomorrow
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # for the day after tomorrow
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False:
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]     
        return temp.iloc[-48:, :]

# train data
df_train = preprocess_data(train) # (52464, 9) 7 + 2
from sklearn.model_selection import train_test_split
x1_train, x1_eval, y1_train, y1_eval = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.2, random_state=0) # Target 1
x2_train, x2_eval, y2_train, y2_eval = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.2, random_state=0) # Target 2
# x -> (samples, 7), y -> (samples, 2)

# test data
df_test = []
for i in range(81):
    file_path = 'C:/data/csv/dacon/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)
x_test = pd.concat(df_test)

#2. model
#3. compile and fit
#4. evaluate and predict
def run_model(x_train, x_eval, y_train, y_eval):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
    model = Sequential()
    model.add(SimpleRNN(32, activation='relu', input_shape=(x_train[1],)))
    model.add(Dropout(0.5))
    nodes = [16, 8, 4, 1]
    for i in nodes:
        model.add(Dense(i))
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    rlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

    from tensorflow.keras.optimizers import Adam
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[es, rlr], validation_split=0.2)
    
    loss = model.evaluate(x_eval, y_eval)
    pred = pd.Series(model.predict(x_eval).round(2))
    return loss, pred

# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
results_1.sort_index()[:48]
# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
results_2.sort_index()[:48]

results_1.sort_index().iloc[:48]
results_2.sort_index()
print(results_1.shape, results_2.shape)
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
submission
submission.iloc[:48]
submission.iloc[48:96]
submission.to_csv('./data/submission_v3.csv', index=False)
