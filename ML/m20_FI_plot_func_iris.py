from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features) # y축의 한계 설정
    return plt

#1. data
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=44)

#2. model
model = DecisionTreeClassifier(max_depth=4)

#3. fit
model.fit(x_train, y_train)

#4. score and predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

#5. visualization
plot_feature_importances_dataset(model)
plt.show()

# [0.00787229 0.         0.4305627  0.56156501]
# acc : 0.9333333333333333


