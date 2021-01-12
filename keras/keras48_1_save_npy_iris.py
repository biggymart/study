from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()

# Dictionary
# (1) {key: value} pair, (2) not ordered, (3) flexible regarding value datatype

# print(dataset)
# {'data': array([[5.1, 3.5, 1.4, 0.2], ...), 'target': array([0,0], ...), ...}
# iris 데이터는 csv 파일로 들어가있다

# print(dataset.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

x_data = dataset.data
y_data = dataset.target
# x_data = dataset['data'] # 딕셔너리 용법, key는 string
# y_data = dataset['target']

print(dataset.frame)
print(dataset.target_names) # ['setosa' 'versicolor' 'virginica']
print(dataset["DESCR"])
print(dataset["feature_names"])
print(dataset.filename)

print(type(x_data), type(y_data)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('../data/npy/iris_x.npy', arr=x_data) # numpy extension: .npy
np.save('../data/npy/iris_y.npy', arr=y_data)