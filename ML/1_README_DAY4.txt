2021-02-01
range: m24 ~ 34

keywords: 
(1) XGB
(2) PCA (Feature engineering, 불필요 칼럼 제거"차원축소"로 속도 향상)

review on last week
kfold, cross_val_score, GridSearchCV, RandomizedSearchCV, Feature_importances (feature engineering), pipeline, 트리계열 DT, RF, GB, XGB, LGBM

XGB 느림, 성능은 좋음 -> 느려서 나온 게 (Light Gradient Boosting Machine)

PCA (비지도 학습에서 쓰임, 차원축소, Principal Component Analysis)
FI 의 비중에 따라 feature를 줄일지 고려해야
FI가 비슷비슷하면 압축이 가능한데 4000개였던 것이 특성을 뽑아내서 200개로 준다면 그리고 성능이 같다면 굿굿
col 너무 많은데 그 모든 피처에 대해서 돌리는 건 자원낭비, 피처를 줄여야 할 필요가 있음
딥러닝의 Dropout과 비견됨

FI와 PCA는 구분해야
FI는 원데이터에 대해서 변형이 없음
PCA는 원데이터에 대한 압축이 되어서 변형이 있음 (전처리의 일부)

xgboost라는 강력한 아이를 배웠다 (결과가 잘 나오면 그냥 그렇게 한 거고, 못 나온 거면 내가 뭔가 잘못한 거다, 파라미터 튜닝할 것 별로 없음)

숙제:
f score
XGB 파라미터 정리