# %% src url: https://dacon.io/competitions/official/235697/codeshare/2361?page=1&dtype=recent&ptype=pub
from IPython import get_ipython

# 이번 시간에서는 Test Time Augmentation(TTA) 기법에 대해 알아보는 시간을 가져보겠습니다.
# 해당 노트북은 (https://www.kaggle.com/andrewkh/test-time-augmentation-tta-worth-it)을 번역 및 참고하여 만들었습니다.
# ## What is Test Time Augmentation ?
# TTA는 말 그대로 Inference(Test) 과정에서 Augmentation 을 적용한 뒤 예측의 확률을 평균(또는 다른 방법)을 통해 도출하는 기법입니다. 모델 학습간에 다양한 Augmentation 을 적용하여 학습하였을시, Inference 과정에서도 유사한 Augmentation 을 적용하여 정확도를 높일 수 있습니다. 또는 이미지에 객체가 너무 작게 위치한 경우 원본 이미지, Crop 이미지를 넣는 등 다양하게 활용이 가능 합니다. 
# (https://preview.ibb.co/kH61v0/pipeline.png)
# 아래는 참고 노트북에서 Dogs vs Cats Task에 TTA를 활용한 예제입니다. 
# 시간이 된다면 컴퓨터 비전 경진대회에도 TTA 를 적용한 코드를 업데이트 하겠습니다. 아래의 노트북에서는 edafa 라는 라이브러리를 사용하였지만, 해당 라이브러리를 사용하지 않고도 적용 가능합니다.
# ## A proof of concept on Dogs vs. Cats competition

# %%
# important dependencies
from sklearn.model_selection import train_test_split
#import keras
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model
import numpy as np
import random
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
###1. (data) Let's explore the dataset!

TRAIN_DIR = './cat-dog/train/'
print ('This is a sample of the dataset file names')
print(os.listdir(TRAIN_DIR)[:5])

sample = plt.imread(os.path.join(TRAIN_DIR,random.choice(os.listdir(TRAIN_DIR))))
print ('Visualize a sample of the image')
print ('Image shape:',sample.shape)
plt.imshow(sample)

# %% [markdown]
###1-0. preprocessing
# Split data into train and validation sets (70% and 30% respectively)
f_train, f_valid = train_test_split(os.listdir(TRAIN_DIR), test_size=0.7, random_state=42)

# Network input size
PATCH_DIM = 32

# ### Build data generator that reads batch by batch from disk when needed
# src: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, files, batch_size=32, dim=(224,224), n_channels=3, n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.random.choice(len(self.files), self.batch_size)
        
        # Find files of IDs
        batch_files = self.files[indexes]

        # Generate data
        X, y = self.__data_generation(batch_files)

        return X, y

    def __data_generation(self, batch_files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, f in enumerate(batch_files):
            # Store sample
            img_path = os.path.join(TRAIN_DIR,f)
            img = image.load_img(img_path, target_size=self.dim)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = np.squeeze(x)
            X[i,:,:,:] = x

            # Store class
            if 'dog' in f:
                y[i]=1
            else:
                y[i]=0
                
        return X, y

training_generator = DataGenerator(np.array(f_train), dim=(PATCH_DIM, PATCH_DIM))

# %%
# ###2. (model) Build and train the model
# Our model is reusing VGG16 architecture without the fully connected layers. So we used the weights from imagenet and add our head as shown

initial_model = VGG16(weights="imagenet", include_top=False, input_shape = (PATCH_DIM, PATCH_DIM, 3))
last = initial_model.output

x = Flatten()(last)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(initial_model.input, preds)

# %% [markdown]
###3. (compile and fit) Now we train the model
model.compile(Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(generator=training_generator,
                    use_multiprocessing=True,
                    epochs=1,
                    workers=8)

# %% [markdown]
###4. (evaluate and predict) Read validation images and evaluate the model

X_val = np.empty((len(f_valid), PATCH_DIM, PATCH_DIM ,3))
y_val = np.empty((len(f_valid)), dtype=int)

for i, f in enumerate(f_valid):
    # Store sample
    img_path = os.path.join(TRAIN_DIR,f)
    img = image.load_img(img_path, target_size=(PATCH_DIM,PATCH_DIM))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    X_val[i,:,:,:] = x

    # Store class
    if 'dog' in f:
        y_val[i]=1
    else:
        y_val[i]=0

y_pred = model.predict(X_val)

y_pred = [(y[0]>=0.5).astype(np.uint8) for y in y_pred]

print('Accuracy without TTA:', np.mean((y_val==y_pred)))

