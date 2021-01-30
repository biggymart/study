from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. data
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

#2. model
model = DecisionTreeRegressor(max_depth=4) ### Takeaway1 ###

#3. fit
model.fit(x_train, y_train)

#4. score and predict
acc = model.score(x_test, y_test)

print(model.feature_importances_) ### Takeaway2 ###
print("acc :", acc)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [3.09767103e-02 0.00000000e+00 3.19268150e-01 2.16063823e-04
#  1.83192445e-02 6.06279792e-02 0.00000000e+00 0.00000000e+00
#  5.70591853e-01 0.00000000e+00]
# acc : 0.2979755178635194

