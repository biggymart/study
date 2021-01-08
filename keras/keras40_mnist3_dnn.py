# CNN으로 들어갈 수 있게 reshape: x.shape = (60000, 28, 28) --> (x.shape[1], x.shape[2], 1)
# DNN으로 들어갈 수 있게 reshape: x.shape = (60000, 28, 28) --> ((x.shape[1]*x.shape[2]),)
# x.train.shape=(N, 28, 28) = (N, 28*28) = (N, 764)
# DNN 모델로 구성 가능

# 주말과제> Dense 모델로 구성, input_shape=(28*28,)