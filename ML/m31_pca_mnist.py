# Task requirement:
# By using PCA, find out how many columns show evr above 0.95.
# Solution: reshape into 2D -> dataset EDA -> set n_components

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)

x_2d = np.reshape(x, (-1, 28 * 28))
pca = PCA(n_components=154)

'''
# dataset EDA to determine n_componenets
pca = PCA()
pca.fit(x_2d)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print("d :", d) 
# d : 154
'''
