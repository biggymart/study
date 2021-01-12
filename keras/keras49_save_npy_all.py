import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

#1-1. sklearn datasets
# boston
boston_dataset = load_boston()
boston_x = boston_dataset.data
boston_y = boston_dataset.target

np.save('../data/npy/boston_x.npy', arr=boston_x)
np.save('../data/npy/boston_y.npy', arr=boston_y)

# diabetes
diabetes_dataset = load_diabetes()
diabetes_x = diabetes_dataset.data
diabetes_y = diabetes_dataset.target

np.save('../data/npy/diabetes_x.npy', arr=diabetes_x)
np.save('../data/npy/diabetes_y.npy', arr=diabetes_y)

# breast_cancer
cancer_dataset = load_breast_cancer()
cancer_x = cancer_dataset.data
cancer_y = cancer_dataset.target

np.save('../data/npy/cancer_x.npy', arr=cancer_x)
np.save('../data/npy/cancer_y.npy', arr=cancer_y)

# iris (생략, keras48에서 이미 함)

# wine
wine_dataset = load_wine()
wine_x = wine_dataset.data
wine_y = wine_dataset.target

np.save('../data/npy/wine_x.npy', arr=wine_x)
np.save('../data/npy/wine_y.npy', arr=wine_y)

#1-2. keras datasets
# mnist
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
np.save('../data/npy/mnist_x_train.npy', arr=mnist_x_train)
np.save('../data/npy/mnist_y_train.npy', arr=mnist_y_train)
np.save('../data/npy/mnist_x_test.npy', arr=mnist_x_test)
np.save('../data/npy/mnist_y_test.npy', arr=mnist_y_test)

# fashion_mnist
(fashion_mnist_x_train, fashion_mnist_y_train), (fashion_mnist_x_test, fashion_mnist_y_test) = fashion_mnist.load_data()
np.save('../data/npy/fashion_mnist_x_train.npy', arr=fashion_mnist_x_train)
np.save('../data/npy/fashion_mnist_y_train.npy', arr=fashion_mnist_y_train)
np.save('../data/npy/fashion_mnist_x_test.npy', arr=fashion_mnist_x_test)
np.save('../data/npy/fashion_mnist_y_test.npy', arr=fashion_mnist_y_test)

# cifar10
(cifar10_x_train,cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()
np.save('../data/npy/cifar10_x_train.npy', arr=cifar10_x_train)
np.save('../data/npy/cifar10_y_train.npy', arr=cifar10_y_train)
np.save('../data/npy/cifar10_x_test.npy', arr=cifar10_x_test)
np.save('../data/npy/cifar10_y_test.npy', arr=cifar10_y_test)

# cifar100
(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()
np.save('../data/npy/cifar100_x_train.npy', arr=cifar100_x_train) 
np.save('../data/npy/cifar100_y_train.npy', arr=cifar100_y_train)
np.save('../data/npy/cifar100_x_test.npy', arr=cifar100_x_test)
np.save('../data/npy/cifar100_y_test.npy', arr=cifar100_y_test)
