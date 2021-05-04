# binary classifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,\
  ModelCheckpoint
import autokeras as ak

#1. dataset
datasets = load_breast_cancer()
x = datasets.data # (569, 30)
y = datasets.target # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

#2. modeling
clf = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=2
)

#3. fit
es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
# ck = ModelCheckpoint('./temp/', save_best_only=True,
#   save_weights_only=True, monitor='val_loss', verbose=1
# )

clf.fit(x_train, 
        y_train,
        epochs=10,
        validation_split=0.2,
        callbacks=[es, lr]
)

#4. evaluate
results = clf.evaluate(x_test, y_test)
print(results)

#5. save model
model2 = clf.export_model()
try:
  model2.save('./keras/keras3/save/model2_cancer/', save_format='tf')
except Exception:
  model2.save('model2_cancer.h5')

best_model = clf.tuner.get_best_model()
best_model.save("./keras/keras3/save/best_model_cancer/", save_format='tf')
