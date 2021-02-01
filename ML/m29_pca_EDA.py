# PCA의 EDA (Exploratory Data Analysis)
# evr과 np.cumsum을 활용
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA


datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA() # n_components 없이
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) # 가장 작은 값부터 누적하여 합한다
print("cumsum :", cumsum)
# n_components :     1          2          3          4          5          6          7          8          9 10
# cumsum : [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]

d = np.argmax(cumsum >= 0.95) + 1
print("cumsum >= 0.95 :", cumsum >= 0.95)
print("d :", d)
# cumsum >= 0.95 : [False False False False False False False  True  True  True]
# d : 8

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()