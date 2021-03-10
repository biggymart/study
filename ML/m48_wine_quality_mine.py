import pandas as pd
import numpy as np

wine_df = pd.read_csv('C:/data/csv/winequality-white.csv', sep=';')

'''
########## EDA ##############
print(wine_df)
print(wine_df.columns)
# [4898 rows x 12 columns]
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')
print(np.unique(wine_df.iloc[:,11]))
# [3 4 5 6 7 8 9] # 퀄리티 종류

# df_check = wine_df.isnull().values.any()
# print(df_check)
# False # 결측치 없음

#############################
'''

wine_np = wine_df.to_numpy()
# print(wine_np)
# print(wine_np.shape)

X = wine_np[:, :-1]
y = wine_np[:, -1]
y = y.astype(np.int64)
# print(X.shape) # (4898, 11)
# print(y.shape) # (4898,)
# print(y)
# print(np.unique(y))

# 7 종류 classification
# 전처리

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.tree import ExtraTreeClassifier

lb = LabelBinarizer()
y = lb.fit_transform(y)
# print(y[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=66
)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#############################################

'''
####### modeling ###########
clf = ExtraTreeClassifier()
clf.fit(X_train, y_train)

aaa = clf.score(X_test, y_test)
print("model.score :", aaa)
############################
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(11,)))
model.add(Dense(20))
model.add(Dense(7, activation='softmax'))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
re = ReduceLROnPlateau(patience=5, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, epochs=1000, batch_size=4, callbacks=[es, re], validation_split=0.2)

#4. evaluate and predict
loss = model.evaluate(X_test, y_test, batch_size=4)
print("loss :", loss)

y_pred = model.predict(X_test)
print("y_predict :", y_pred)





