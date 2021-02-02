2021-02-02
range: m35 ~ m41

keywords:
(1) eval_metric, eval_set, evals_result
(2) early_stopping_rounds
(3) pickle, joblib, xgb model save

REVIEW
pca를 통과해나온 값은 y값
1. 차원축소
2. 노이즈 제거 탁월 (강한 특성만 뺌)

m35의 중점은 
(1) fit 부분의 eval_metric, eval_set 파라미터, 그리고
(2) 최종적으로 결과를 보기 위해 print하는 evals_result 메소드이다

특히 eval_metric에 여러가지 metrics를 넣을 수 있다는 점을 보기 위해서
m36에서 리스트를 활용하여 fit 부분에 eval_metric를 수정하였다.

m37은 earlystopping에 관한 내용이다.
fit 부분에 early_stopping_rounds 이란 파라미터만 추가해주면 된다.

m38에서는 시각화한다.
특히 early_stopping 의 기준이 eval_metric의 마지막 요소임을 기억하자.

m39 ~ m41은 모델을 저장(model, weight save)하는 3가지 방식을 다룬다. (한 파일로 압축함)
python에서 제공하는 기능인 pickle과 joblib, xgb 자체의 model save가 있다.

xgboosting의 각종 기능
early_stopping, model save