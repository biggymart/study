# evr 0.95, 1.0
# gridSearch, randomizedSearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import numpy as np

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255. # 784
x_test = x_test.reshape(10000, 28 * 28)/255.

n_compo = 154 # 713 for evr >= 1.0
pca = PCA(n_components=n_compo)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

#2. model
parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.001, 0.01, 0.1, 0.3], "max_depth" : [4, 5, 6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.001, 0.01, 0.1], "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate" : [0.001, 0.1, 0.5], "max_depth": [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]

seed = 44
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)
model_search = GridSearchCV(model, parameters, cv=kfold) # n_estimator == epochs

#3. fit
model_search.fit(x_train_pca, y_train, eval_metric='mlogloss', verbose=True, eval_set=[(x_train_pca, y_train), (x_test_pca, y_test)], early_stopping_rounds=3)
results = model_search.score(x_test_pca, y_test)
print(results)

model.save_model("../data/xgb_save/m34.xgb.model")

# n_compo = 154; GridSearchCV
# n_compo = 713; GridSearchCV

# n_compo = 154; RandomizedSearchCV
# n_compo = 713; RandomizedSearchCV

