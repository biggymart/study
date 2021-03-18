import tensorflow as tf

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  TRAIN_DIR,
  validation_split=VAL_SPLIT,
  subset="training",
  seed=SEED,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)
# image_dataset_from_directory를 이용해서 해당 디렉토리에서 이미지를 가져오기

# 학습과 검증 데이터를 나눔.

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  TRAIN_DIR,
  validation_split=VAL_SPLIT,
  subset="validation",
  seed=SEED,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)
# 클래스 이름을 가져오기

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  TEST_DIR,
  seed=SEED,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  shuffle=False)
# 클래스 이름을 가져오기

class_names = train_dataset.class_names
print(class_names)
# 확인해보면 ['0~999']
# 이라고 하위 폴더명이 바로 클래스 레이블

 

for image_batch, labels_batch in train_dataset:
  print(image_batch.shape)
  # 32, 224, 224, 3
  print(labels_batch.shape)
  # 32
  break
# 이미지의 shape를 알아보려면 위와 같이

 

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)