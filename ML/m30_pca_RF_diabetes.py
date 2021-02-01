# PCA의 적용; model = RandomForest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

#1. data
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

# evr = 0.95
pca = PCA(n_components=8)
x_pca = pca.fit_transform(x)

#2. model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()