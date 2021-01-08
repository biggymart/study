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
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(10,10,1))) # 10 x 10, gray_scale
# batch_size는 아직 알 필요 없지, 기계 만들 때 몇개 들어가야 하는지 정해야 하니? 걍 이 크기만 맞으면 들여보내 줘
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
# 2차원으로 잘 나오는 것을 알 수 있다. (N, 5, 5, 1)이 (N, 25)로 평평해진 것임