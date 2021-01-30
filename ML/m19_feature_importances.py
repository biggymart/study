from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. data
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=44)

#2. model
model = DecisionTreeClassifier(max_depth=4) ### Takeaway1 ###

#3. fit
model.fit(x_train, y_train)

#4. score and predict
acc = model.score(x_test, y_test)

print(model.feature_importances_) ### Takeaway2 ###
print("acc :", acc)

# [0.         0.00787229 0.96203388 0.03009382] 
# 총합은 1, 첫번째 두번째 feature는 중요도 낮음 (없어도 좋아, 단, DecisionTreeClassifier 이라는 모델에 한정해서)
# 과적합의 문제, 자원을 잡아먹는 문제
# acc : 0.9333333333333333
