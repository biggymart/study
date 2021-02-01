# PCA 개념설명 "차원축소"
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA


datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=9)
x2 = pca.fit_transform(x)
print(x2.shape) # (442, 7) # 컬럼의 수 재구성

# 압축했을 때 어느 부분이 중요한지 알려줌
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
# n_components=7 : 0.9479436357350414
# n_components=8 : 0.9913119559917797
# n_components=9 : 0.9991439470098977
# 각 col의 중요도 표시, 모든 col의 합은 압축률을 나타내줌 (pca에서 뽑아낸 피처의 특성 정도, 통상 90퍼 이상이면 오케이)