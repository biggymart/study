# save했으니까 load해봐야겠지?
# 필요한 api부터 임포트하자

from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')

model.summary()
# summary가 잘 되면 잘 불러져왔다는 의미겠지

# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
# 컴파일을 안 했다는 경고임, 가중치 저장할 때 다시 확인할 예정