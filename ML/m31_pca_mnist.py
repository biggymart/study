# Task requirement:
# By using PCA, find out how many columns show evr above 0.95.
# Solution: reshape into 2D -> dataset EDA -> set n_components

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x = np.append(x_train, x_test, axis=0)
# step1. print(x.shape) -> check the shape of data : (70000, 28, 28)
# step2. reshape into 2D
x_2d = np.reshape(x, (-1, 28 * 28)) # 784 columns
# step3. dataset EDA to determine n_componenets

evr_cutoff = 1.0 # customize this value
pca = PCA()
pca.fit(x_2d)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= evr_cutoff) + 1
print("d :", d) 
# ever_cutoff : 0.95 -> d : 154
# ever_cutoff : 1.0  -> d : 713

# step4. use the d
pca = PCA(n_components=154)


