# keras22_1_iris1_keras.py Copy and Paste
# Study objective: Able to compare the overall structure (and terms) btwn DL and ML
# ML: data -> model -> fit -> score and predict

#1. data
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler() # or Standardscaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

# ML does not need OneHotEncoding ### Takeaway1 ###
# OneHotEncoding, tensorflow
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

from sklearn.svm import LinearSVC # In later lessons, other powerful models will be used (such as LGBM)
model = LinearSVC() # Compared to the lengthy lines of DL, ML only uses one line (super concise!) ### Takeaway2 ###


#3. fit
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=10000)

model.fit(x_train, y_train) # Again, ML is concise


#4. score and predict
# result = model.evaluate(x_test, y_test)
# print(result)

result = model.score(x_test, y_test) # Instead of 'evaluate', use 'score' ### Takeaway3 ###
print(result) # ML immediately returns accuracy (even w/o explicit notation), whereas DL returns loss and metrics
# DL required much effort to increase accuracy // ML in general has a good output even in a default setting

y_pred = model.predict(x_test[-5:-1]) # ML uses the same term 'predict'
print(y_pred) 
