from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense

inp1 = Input((10,)) # scalar 10 인 vector
inp2 = Input((20,)) # scalar 20 인 vector

lmb_layer = Lambda(lambda x: [x/2, x*2])

a1, b1 = lmb_layer(inp1) # first layer => layer index is [0], creates _inbound_nodes[0]
a2, b2 = lmb_layer(inp2) # second layer => layer index is [1], creates _inbound_nodes[1]

print(lmb_layer._inbound_nodes)
# [<tensorflow.python.keras.engine.node.Node object at 0x0000026226C26880>, <tensorflow.python.keras.engine.node.Node object at 0x0000026226C4D1F0>]
print(lmb_layer._inbound_nodes[0].output_tensors)
# [<KerasTensor: shape=(None, 10) dtype=float32 (created by layer 'lambda')>, <KerasTensor: shape=(None, 10) dtype=float32 (created by layer 'lambda')>]
# corresponds to a1, b1, respectively
print(lmb_layer._inbound_nodes[1].output_tensors)
# [<KerasTensor: shape=(None, 20) dtype=float32 (created by layer 'lambda')>, <KerasTensor: shape=(None, 20) dtype=float32 (created by layer 'lambda')>]
# corresponds to a2, b2, respectively

d1 = Dense(10)(a1)
d2 = Dense(20)(b1)
d3 = Dense(30)(a2)
d4 = Dense(40)(b2)

model = Model(inputs=[inp1, inp2], outputs=[d1, d2, d3, d4])
model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, 10)]         0
# __________________________________________________________________________________________________
# input_2 (InputLayer)            [(None, 20)]         0
# __________________________________________________________________________________________________
# lambda (Lambda)                 multiple             0           input_1[0][0]
#                                                                  input_2[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 10)           110         lambda[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 20)           220         lambda[0][1]
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 30)           630         lambda[1][0]
# __________________________________________________________________________________________________
# dense_3 (Dense)                 (None, 40)           840         lambda[1][1]
# ==================================================================================================
# Total params: 1,800
# Trainable params: 1,800
# Non-trainable params: 0
# __________________________________________________________________________________________________

'''
layer_name[x][y]
1. layer_name corresponds to the layer where the input tensors of this layer comes from
2. x : node index
3. y : tensor index
'''