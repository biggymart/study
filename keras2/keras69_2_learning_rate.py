# 실제값
x_train = 0.5
y_train = 0.8

# 훈련 파라미터
weight = 0.5
lr = 0.01
epoch = 100

# 모델링 (MSE)
for iteration in range(epoch):
    y_predict = x_train * weight
    error = (y_train - y_predict) **2 # loss = cost = error

    print("Error : {:6.5f} \ty_predict : {:6.5f}".format(error, y_predict))

    up_y_predict = x_train * (weight + lr)
    up_error = (y_train - up_y_predict) ** 2

    down_y_predict = x_train * (weight - lr)
    down_error = (y_train - down_y_predict) ** 2

    # 더 작은 에러를 골라서 가중치를 업데이트
    if(down_error <= up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr


'''
iteration: 1
y_predict = 0.5 * 0.5 = 0.25
error = (0.8 - 0.25) ** 2 = 0.3025

up_y_predict = 0.5 * (0.5 + 0.01) = 0.255
up_error = (0.8 - 0.255) ** 2 = 0.297025

down_y_predict = 0.5 * (0.5 - 0.01) = 0.245
down_error = (0.8 - 0.245) ** 2 = 0.308025

up_error < down_error:
    weight = 0.5 + 0.01 = 0.51


'''
