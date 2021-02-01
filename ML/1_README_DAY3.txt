2021-01-29
range: m14 ~ m23

keywords: Pipeline, make_pipeline (+ grid/randomizedSearch), Feature Importances (fit 이후에 가능)

오늘은 Tree계열을 배우는 것
뭔가 실습이 많은 느낌이다... 벅차군

1교시
머신러닝 전체에 대해서 리뷰를 좀 해보자.
첫 인상: 겁나 빠르고 성능도 중상타 하네~ (쌔빠지게 튜닝 한 것 보다 낫네)
XOR 문제, DNN으로 해결 (30년 전 장비가 못 따라줘서 미친 짓이었음)
iris로 대표적인 모델들을 봤다. 여러 DecisionTree가 엮인 게 RandomForest임 (ensemble)
분류 -> Classifier (accuracy); 회귀 -> Regressor (r2)
머신러닝은 4단계로 구성 data -> model -> fit -> score and predict (score는 evaluate와 비유됨)

'모든 모델 확인'
all_estimators: 모든 모델을 뽑아서 평가를 해봤다, 요구하는 값과 불일치하는 것을 처리하기 위해서 예외처리 (try, except) 배움

'train_test_split의 진화'
KFold 목적: validation하는데 전체적으로 다 하는 것보다 예를 들어 5등분으로 하여 구간마다 검증
cross_val_score에 모델, 데이터, cv=kfold로 잡아준다; 이름에 score 들어가 있으니까 fit도 되는 거 알겠지
all_estimators에 kfold를 적용했고, 시간이 많이 걸리지 않았다

'파라미터의 자동화'
본격적인 활용도가 높아지는 부분은 GridSearch부터다; 파라미터 list of dict로 만들어주고, kfold도 적용해줬음. 그리고 RandomForest 모델을 썼음.
파라미터에 대해서 일부 설명 들어갔음 (depth, n_jobs), 파라미터를 직렬로 이었음 (곱셈)
GridSearch의 문제점은 너무 많이 돌아가서 시간 소비가 컸음 -> RandomizedSearchCV 등장 (지금은 그나마 빠르지만 나중에 keras에 엮으면 느려짐)
n_iter (default: 10) * cross_val_score = total cycle

오늘은 전처리 들어감.

2교시
'전처리와 모델 이어주기'
from sklearn.pipeline import Pipeline, make_pipeline
scaler와 model을 엮어주는 것이 pipeline이다

이후, scaler + model + GridSearch

3교시
실습: m16

4교시
test set을 사용할 수 있는 다른 방법 없을까?
nested cv

5교시
오후에는 중요한 거 한다고 했지
이제부터는 중요한 거 할거니까 정신 바짝 차려라

6교시
plot 해보고 feature importances 파악하고 중요하지 않은 feature를 줄여라

7교시
m21, 22 실습
