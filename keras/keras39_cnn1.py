# CNN을 주로 쓰는 곳은 이미지다 (object detection)
# 컴퓨터는 전부를 보는 게 아니라 하나하나씩 수치화(RGB)된 픽셀로 본다
# 사람 이미지를 주면, 얼굴이 특성이 가장 많고(높은 가중치), 몸이 배경보다는 특성이 좀 더 많다

# 조각조각 내서 특성을 뽑아낸다
# 속성을 찾기 위해서 데이터 증폭했다가 줄임, 1by1, 2by2, 3by3
# (인풋노드는 하나인데 히든노드가 많아지고 아웃풋노드는 하나로 줄어드는 것과 유사)
# X는 이미지, Y는 사람 (이미지가 사람인지 구분하는 모델)

# 과제> keras.io 에 들어가서 conv2D 를 들어가서 각 파라미터의 이름을 찾아라
# Conv2D(10, (2,2), input_shape=(5,5,1)) 
# input_shape=(rows, cols, channels) if data_format='channels_last' (default)
# filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
# kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.

# input_shape 쉽게 설명하면, (1) 노드 갯수, (2) 조각(x by y), (3) 전체 데이터 shape과 색(흑백은 1, 컬러는 3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1,
                 padding='same', input_shape=(10,10,1))) # 10 x 10, gray_scale
# batch_size는 아직 알 필요 없지, 기계 만들 때 몇개 들어가야 하는지 정해야 하니? 걍 이 크기만 맞으면 들여보내 줘
model.add(MaxPooling2D(pool_size=2)) # default=2, 3은 3개로 쪼개기, (2,3) 가로 세로 다르게 쪼갤 수 있음
model.add(Conv2D(9, (2,2), padding='valid')) # 1레이어가 4차원 받아서 4차원 출력하니까 2레이어가 입력받는 데 문제 없음
# model.add(Conv2D(9, (2,3)))
# model.add(Conv2D(8, 2))
model.add(Flatten())
model.add(Dense(1))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
dense (Dense)                (None, 9, 9, 1)           11
=================================================================
Total params: 61
Trainable params: 61
Non-trainable params: 0
_________________________________________________________________
'''
# 아니, 우리는 사람인지 아닌지만 구분하고 싶은 건데, Dense 레이어가 4차원을 뽑아내내? 뭔가 문제가 심각하다..

# (N, 5, 5, 1) --> Conv2D(10, (2,2)) --> (4, 4, 10)
# (N, 25) 해도 내용이 바뀌진 않지, 평평하게 폈지? flatten 작업
# Conv2D 레이어와 Dense 레이어를 잇기 위해선 model.add(Flatten())을 써야 한다

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
flatten (Flatten)            (None, 810)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 811
=================================================================
Total params: 861
Trainable params: 861
Non-trainable params: 0
_________________________________________________________________
'''
# 2차원으로 잘 나오는 것을 알 수 있다. (N, 5, 5, 1)이 (N, 25)로 평평해진 것임 (한 행 한 행 짤라서 한 줄로 나열하기)
# Conv2D는 4차원을 받아서 4차원으로 출력한다

# 2개 이상의 Conv2D 레이어를 쌓을 수 있다
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 8)           296
_________________________________________________________________
flatten (Flatten)            (None, 392)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 393
=================================================================
Total params: 1,108
Trainable params: 1,108
Non-trainable params: 0
_________________________________________________________________
'''
# Conv2D은 특성을 추출하는 것이라서 LSTM과 다르게 2개 이상을 쌓는다고 해서 성능이 떨어지는 것은 아니다
# param_number = output_channel_number * (input_channel_number * kernel_height * kernel_width + 1)
# 10 * (1 * 2 * 2 + 1) = 50

# Cf> Dense 레이어의 parameter number 구하기 공식
# param_number = output_channel_number * (input_channel_number + 1)
# https://towardsdatascience.com/how-to-calculate-the-number-of-parameters-in-keras-models-710683dae0ca

# 크기를 유지시키고 싶으면 padding='same'
# 가장자리에 잘려나가는 것 방지하기 위해서
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 9)           369
_________________________________________________________________
flatten (Flatten)            (None, 729)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 730       
=================================================================
Total params: 1,149
Trainable params: 1,149
Non-trainable params: 0
_________________________________________________________________
'''
# shape이 안 줄어든 것을 확인할 수 있다. (인풋 들어온 크기로 아웃풋 나감)

# strides : 보폭
# 자르기 위해 건너뛰는 폭을 조절할 수 있음 (default는 1), 가로세로 다르게 할 수 있음 (e.g. strides=(1,2))
# 겹치는 게 좀 더 특성을 잘 뽑아내는데 두번째 레이어부터 판단해보길, 데이터마다 다름

# MaxPooling2D 적용 후
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 5, 5, 10)          0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 4, 4, 9)           369
_________________________________________________________________
flatten (Flatten)            (None, 144)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 145
=================================================================
Total params: 564
Trainable params: 564
Non-trainable params: 0
_________________________________________________________________
'''

# MaxPooling2D(pool_size=2) 일 때
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 5, 5, 10)          0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 4, 4, 9)           369
_________________________________________________________________
flatten (Flatten)            (None, 144)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 145
=================================================================
Total params: 564
Trainable params: 564
Non-trainable params: 0
_________________________________________________________________
'''
