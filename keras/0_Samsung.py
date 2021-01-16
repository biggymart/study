# Interactive CUI version
import numpy as np
import pandas as pd

features = ['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']
target_feature = '시가'
denomination_date = '2018-05-04'
y_col = features.index(target_feature)

#0. basic functions
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

def ask_load_file():
    switch1 = True
    while switch1:
        user_dir = str(input('Enter a full directory of csv : (Enter "quit" to exit) :'))
        try:
            if user_dir == "quit":
                switch1 = False
                print("quit loading file")
                break
            elif user_dir is not False:
                loaded_file = load_df(user_dir)
                return loaded_file
                switch1 = False
                break
            else:
                print("Incorrect input. Try again.")
        except FileNotFoundError:
            print("Could not find the file. Try again.")
            
plus_one = ['등락률', '거래량', '금액(백만)', '신용비',	'개인',	'기관',	'외인(수량)', '외국계',	'프로그램',	'외인비']
def four_plus_one(df, feature):
    idx = plus_one.index(feature)

    df_fix_col = df.loc[:,'시가':'종가']
    df_sel_col = df.loc[:, plus_one[idx]]
    df_alltime = pd.concat([df_fix_col, df_sel_col], axis=1)
    df_after_deno = df_alltime.loc[denomination_date:,:]
    return df_after_deno


def run_model(df, feature):
    df_numpy = df.to_numpy()

    x_npy = df_numpy[:-1, :].astype(np.float64)
    y_npy = df_numpy[1:, y_col].astype(np.float64)

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

    x_pred = df_numpy[-1]
    x_pred = x_pred.reshape(1, -1)
    x_pred = scaler.transform(x_pred)
    x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)
    
    np.savez('../data/npy/samsung_{0}.npz'.format(plus_one.index(feature)), x_test=x_test, y_test=y_test, x_pred=x_pred)
    # npz_loaded = np.load('../data/npy/samsung.npz')
    # x_test = npz_loaded['x_test']
    # y_test = npz_loaded['y_test']
    # x_pred = npz_loaded['x_pred']

    #2. model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout

    model = Sequential()
    model.add(LSTM(300, activation='relu', input_shape=(x_npy.shape[1], 1)))
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
    model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, callbacks=[es, cp], validation_split=0.2)

    model.save('../data/h5/samsung_lstm_{0}.h5'.format(plus_one.index(feature)))

    # from tensorflow.keras.models import load_model
    # model = load_model('../data/h5/samsung_lstm.h5')

    #4. evaluate and predict
    loss = model.evaluate(x_test, y_test, batch_size=32)
    y_pred = model.predict(x_pred, batch_size=32)

    return(loss, y_pred)

### flow control
running = True
while running:
    print('''Title: Four + One
Let's predict a future stock price (종가)!
Program description:
    Step1. Load your first csv file. (MUST enter the full path)    
    Step2. You can optionally merge one or more other csv file(s) with the loaded csv file in Step1.
    Step3. Type in a feature that you want as the fifth variable along with the other four (시가, 고가, 저가, 종가)
Begin the journey...''')
    data1 = ask_load_file()

    switch = True
    loop_count = 0
    while switch:
        ans1 = input("Need more file? (y/n):")
        if loop_count == 0 and ans1 == 'y':
            data2 = ask_load_file()
            data1 = join_dfs(data1, data2)
            loop_count += 1
        elif loop_count > 0 and ans1 == 'y':
            data3 = ask_load_file()
            data1 = join_dfs(data1, data3)
            loop_count += 1
        elif ans1 == 'n':
            print("moving on...")
            break
        else:
            print("Try again")

    # feature selection
    ans2 = input("Select feature (등락률, 거래량, 금액(백만), 신용비, 개인, 기관, 외인(수량), 외국계, 프로그램, 외인비) :")
    df_pandas = four_plus_one(data1, ans2)
    result = run_model(df_pandas, ans2)
    print("loss(mse, mae)", result[0], "predict", result[1])

    switch2 = True
    while switch2:
        ans3 = input("Another round? (y/n):")
        if ans3 == 'y':
            continue
        elif ans3 == 'n':
            quit()
        else:
            print("Try again")