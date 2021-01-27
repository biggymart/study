#1. data (동일)
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

### 분기점 1 #######################################################
# from sklearn.preprocessing import MinMaxScaler, StandardScaler  #
# scaler = MinMaxScaler()                                         # Option 1
# scaler = StandardScaler()                                       # Option 2
# scaler.fit_transform(x_train)                                   #
# scaler.transform(x_test)                                        #
###################################################################

### 분기점 2 #############################################
# DL                                                    # Option 1
# from tensorflow.keras.utils import to_categorical     # /
# y_train = to_categorical(y_train)                     # /
# y_test = to_categorical(y_test)                       # /
                                                        #
# ML                                                    # Option 2
# nothing                                               #
#########################################################

### 분기점 3 #####################################################
#2. model                                                       #
# DL                                                            #
# from sklearn.neighbors import KNeighborsClassifier            # Option 1
# model = KNeighborsClassifier()                                #
# from sklearn.tree import DecisionTreeClassifier               # Option 2 
# model = DecisionTreeClassifier()                              #
# from sklearn.ensemble import RandomForestClassifier           # Option 3
# model = RandomForestClassifier()                              #
                                                                #
# from tensorflow.keras.models import Sequential                # All DL
# from tensorflow.keras.layers import Dense                     # /
# model = Sequential()                                          # /
# model.add(Dense(10, activation='relu', input_shape=(4,)))     # /
# model.add(Dense(10))                                          # /
# model.add(Dense(10))                                          # /
# model.add(Dense(3, activation='softmax'))                     # /
                                                                #
# ML                                                            #
# from sklearn.svm import LinearSVC, SVC                        # Option 4
# model = LinearSVC()                                           #
# model = SVC()                                                 # Option 5
#################################################################

### 분기점 4 #############################################################################
#3. compile and fit                                                                     #
# DL                                                                                    #
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])     # Option 1
# model.fit(x_train, y_train, epochs=100)                                               # /
                                                                                        #
# ML                                                                                    #
# model.fit(x_train, y_train)                                                           # Option 2
#########################################################################################

### 분기점 5 #####################################
#4. score and predict                           #
# DL                                            #
# result = model.evaluate(x_test, y_test)       # Option 1
# print(result)                                 # /
                                                #
# ML                                            #
# result = model.score(x_test, y_test)          # Option 2
# y_pred = model.predict(x_test)                # /
                                                # /
# from sklearn.metrics import accuracy_score    # /
# acc = accuracy_score(y_test, y_pred)          # /
# print("accuracy_score :", acc)                # /
#################################################


### 결과 ###
# 1. MinMaxScaler
#   1-1. DL
#       1-1-1. KNeighborsClassifier
#       [0.13101902604103088, 0.9666666388511658]

#       1-1-2. DecisionTreeClassifier
#       [0.3745329678058624, 0.9333333373069763]

#       1-1-3. RandomForestClassifier
#       [0.3589625656604767, 0.9666666388511658]

#   1-2. ML
#       1-2-1. LinearSVC
#       accuracy_score : 0.9666666666666667

#       1-2-2. SVC
#       accuracy_score : 0.9666666666666667

# 2. StandardScaler
#   2-1. DL
#       2-1-1. KNeighborsClassifier
#       [0.3564499616622925, 0.9333333373069763]

#       2-1-2. DecisionTreeClassifier
#       [0.2079426497220993, 0.9666666388511658]

#       2-1-3. RandomForestClassifier
#       [0.14667536318302155, 0.9666666388511658]

#   2-2. ML
#       2-2-1. LinearSVC
#       accuracy_score : 0.9666666666666667

#       1-2-2. SVC
#       accuracy_score : 0.9666666666666667