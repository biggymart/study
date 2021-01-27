#1. data (이중분류)
import numpy as np
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

### 분기점 1 ###
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()  # Option 1
# scaler = StandardScaler() # Option 2
scaler.fit_transform(x_train)
scaler.transform(x_test)

### 분기점 2 ###
#2. model
from sklearn.neighbors import KNeighborsClassifier # Option 1
model = KNeighborsClassifier()
# from sklearn.tree import DecisionTreeClassifier # Option 2 
# model = DecisionTreeClassifier()
# from sklearn.ensemble import RandomForestClassifier # Option 3
# model = RandomForestClassifier()
# from sklearn.linear_model import LogisticRegression # Option 4 (분류)
# model = LogisticRegression()
# from sklearn.svm import LinearSVC # Option 5
# model = LinearSVC()           
# from sklearn.svm import SVC # Option 6
# model = SVC()

#3. fit
model.fit(x_train, y_train)

#4. score and predict
result = model.score(x_test, y_test)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score # 분류
acc = accuracy_score(y_test, y_pred)
print("accuracy_score :", acc)


### 결과 ###
# 1. MinMaxScaler
#   1-1. KNeighborsClassifier
#   accuracy_score : 0.9210526315789473
#   1-2. DecisionTreeClassifier
#   accuracy_score : 0.9122807017543859
#   1-3. RandomForestClassifier
#   accuracy_score : 0.9824561403508771
#   1-4. LogisticRegression
#   accuracy_score : 0.9385964912280702
#   1-5. LinearSVC
#   accuracy_score : 0.8596491228070176
#   1-6. SVC
#   accuracy_score : 0.8947368421052632
# 2. StandardScaler
#   2-1. KNeighborsClassifier
#   accuracy_score : 0.9210526315789473
#   2-2. DecisionTreeClassifier
#   accuracy_score : 0.9298245614035088
#   2-3. RandomForestClassifier
#   accuracy_score : 0.956140350877193
#   2-4. LogisticRegression
#   accuracy_score : 0.9385964912280702
#   2-5. LinearSVC
#   accuracy_score : 0.8245614035087719
#   2-6. SVC
#   accuracy_score : 0.8947368421052632wi