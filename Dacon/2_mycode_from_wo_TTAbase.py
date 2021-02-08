# %% import libraries
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import random
import os
import string
import matplotlib.pyplot as plt

# %%
TRAIN_DIR = 'C:/data/data_2/dirty_mnist_2nd/'

###1-0. preprocessing
f_train, f_valid = train_test_split(os.listdir(TRAIN_DIR), test_size=0.7, random_state=1)

# Network input size
PATCH_DIM = 256

dirty_mnist_answer = pd.read_csv("dirty_mnist_answer.csv")
# dirty_mnist라는 디렉터리 속에 들어있는 파일들의 이름을 namelist라는 변수에 저장
namelist = os.listdir('./dirty_mnist/')

# png to numpy array

# ### Build data generator that reads batch by batch from disk when needed
# 케라스 사용할 거니까 형식은 이걸로
class DatasetMNIST(tensorflow.keras.utils.Sequence):
    'Load dirty Mnist data for Keras'
    def __init__(self, dir_path, meta_df, dim=(256, 256), n_channels=1, n_classes=26):
        self.dir_path = dir_path # 데이터의 이미지가 저장된 디렉터리 경로
        self.meta_df = meta_df # 데이터의 인덱스와 정답지가 들어있는 DataFrame
        self.dim = dim
        self.files = files
        self.n_channels = n_channels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        image = cv2.imread(self.dir_path + str(self.meta_df.iloc[index,0]).zfill(5) + '.png', cv2.IMREAD_GRAYSCALE)
        image = (image/255).astype('float')[..., np.newaxis]

        # 정답 numpy array생성(존재하면 1 없으면 0)
        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

    def __data_generation(self, batch_files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, f in enumerate(batch_files):
            # Store sample
            img_path = os.path.join(TRAIN_DIR, f)
            img = image.load_img(img_path, target_size=self.dim)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = np.squeeze(x)
            X[i,:,:,:] = x
################################ dict이용해서 있는지 없는지 확인해서 해당 열에 1 넣는 게 어떤가
            # Store class
            a2z_lst = list(string.ascii_lowercase)
            for letter in a2z_lst:
                if letter in f:
                    y[a2z_lst.index(letter)]=1 
                else:
                    y[a2z_lst.index(letter)]=0
################################                
        return X, y


# 요런식으로 데이터 불러올 수 있게
    #Dataset 정의
train_dataset = DatasetMNIST("dirty_mnist/", train_answer)
valid_dataset = DatasetMNIST("dirty_mnist/", test_answer)


# %%
# ###2. (model) Build and train the model
initial_model = VGG16(weights="imagenet", include_top=False, input_shape=(PATCH_DIM, PATCH_DIM, 1))
last = initial_model.output

x = Flatten()(last)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.4)(x)
preds = Dense(26, activation='sigmoid')(x) ############################

model = Model(initial_model.input, preds)

# %%
###3. (compile and fit) Now we train the model
model.compile(Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(generator=training_generator,
                    use_multiprocessing=True,
                    epochs=1,
                    workers=8)

# %%
###4. (evaluate and predict) Read validation images and evaluate the model

X_val = np.empty((len(f_valid), PATCH_DIM, PATCH_DIM , 1))
y_val = np.empty((len(f_valid)), dtype=int)

for i, f in enumerate(f_valid):
    # Store sample
    img_path = os.path.join(TRAIN_DIR, f)
    img = image.load_img(img_path, target_size=(PATCH_DIM, PATCH_DIM))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = np.squeeze(x)
    X_val[i,:,:,:] = x

#######################
    # Store class
    if 'dog' in f:
        y_val[i]=1
    else:
        y_val[i]=0
#######################

y_pred = model.predict(X_val)

y_pred = [(y[0]>=0.5).astype(np.uint8) for y in y_pred]

print('Accuracy without TTA:', np.mean((y_val==y_pred)))



