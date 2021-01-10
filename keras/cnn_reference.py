#1. Configure Learning Environment
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import interactive, interact_manual # pip install ipywidgets
# https://ipywidgets.readthedocs.io/en/stable/user_install.html
import tensorflow as tf
from tensorflow.keras import utils, layers, models, losses

#2. Prepare Data
# Download the mnist dataset using keras
(trainDatas, trainLabels), (testDatas, testLabels) = tf.keras.datasets.mnist.load_data()
print("### ORIGINAL ###",
    "\ntrainDatas shape :", trainDatas.shape,
    "\ntrainLabels shape :", trainLabels.shape,
    "\ntestDatas shape :", testDatas.shape,
    "\ntestLabels shape :", testLabels.shape,
    sep=''
)

#3. Preprocessing Data
# 2차원(28*28)이미지를 1차원 배열로 변환
trainDatas = trainDatas.reshape(-1, 28, 28, 1)
testDatas = testDatas.reshape(-1, 28, 28, 1)

print("### AFTER PREPROCESSING ###",
    "\ntrainDatas shape :", trainDatas.shape,
    "\ntestDatas shape :", testDatas.shape,
    sep=''
)

# 레이블 OneHotEncoding
trainLabels = np.eye(10)[trainLabels]
testLabels = np.eye(10)[testLabels]

print("trainLabels shape :", trainLabels.shape,
    "\ntestLabels shape :", testLabels.shape,
    sep=''
)

#4. Generate Model
model = models.Sequential([
    layers.Input([28, 28, 1]),
    layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="valid", activation=tf.keras.activations.relu),
    layers.MaxPool2D((2,2), (2,2)),
    layers.Flatten(),
    layers.Dense(256, activation=tf.keras.activations.relu),
    layers.Dropout(0.5),
    layers.Dense(10, activation=tf.keras.activations.softmax)
])
model.summary()

utils.plot_model(model, 'model.png', show_shapes=True)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

#5. Training
history = model.fit(
    trainDatas, trainLabels,
    batch_size=1000,
    epochs=20,
    validation_split=0.2
)

loss = history.history["loss"]
validationLoss = history.history["val_loss"]
epochs = range(0, len(loss))

plt.plot(epochs, loss, label="Training Loss")
plt.plot(epochs, validationLoss, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#6. Evaluate
evaLoss, evaAcc = model.evaluate(testDatas, testLabels, verbose=0)
print("evaluation loss :", evaLoss, "evaluation accuracy :", "{:3.2f}%".format(evaAcc*100))

#7. Predict
def showTestImage(idx):
    data = testDatas[idx].reshape(-1, 28, 28, 1)
    dataPred = model.predict(data.astype(float))

    plt.imshow(testDatas[idx].reshape(28,28), cmap="gray")
    plt.grid(False)
    plt.title(f"LABEL: {np.argmax(testLabels[idx])}, PREDICT: {np.argmax(dataPred)}")
    plt.show()

widget = interactive(showTestImage, idx=(0, 10000, 1))
widget.display()

#8. Error Find
errsIdx = []
testDatas = testDatas.reshape(-1, 28, 28, 1)
dataPreds = model.predict(testDatas.astype(float))

for idx in range(10000):
    if np.argmax(dataPreds[idx]) != np.argmax(testLabels[idx]):
        errsIdx.append(idx)

print(len(errsIdx), errsIdx)
