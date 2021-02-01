# Task requirement:
# Using n_components ((1) evr >= 0.95 (2) evr >= 1.0) found in m31 file, make a XGBClassifier model (default parameter).
# Its result should be better than the model in keras40_mnist3_dnn.py.
# Compare with CNN model (keras40_mnist2_cnn.py).

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import numpy as np

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28 * 28)/255.

n_compo = 154 # 713 for evr >= 1.0
pca = PCA(n_components=n_compo)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

#2. model
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)

model.fit(x_train_pca, y_train, eval_metric='logloss')

acc = model.score(x_test_pca, y_test)
print("acc :", acc)

# keras40_mnist3_dnn.py results
# [categorical_crossentropy, acc] : [0.1446416974067688, 0.9541000127792358]
# 7 7/2 2/1 1/0 0/4 4/1 1/4 4/9 9/5 6/9 9/

# m33_pca_mnist1_xgb.py
# evr >= 0.95
# acc : 0.1244

# evr >= 1.00
# acc : 0.1352