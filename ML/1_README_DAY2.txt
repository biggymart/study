2021-01-28
range: m10 - m13

keywords: 
(1) KFold, cross_val_score, 
(2) GridSearchCV, parameters, model.best_estimator_, 
(3) RandomizedSearchCV

https://datascienceschool.net/intro.html
https://3months.tistory.com/321
Opening lines.
**CAUTION** What we are about to learn from today is REALLY crucial and it will be THE bedrock knowledge for future learnings.
Yesterday, in m09 files, we looked at all of the models available for both classification and regression. 
However, it is unnecessary and impractical to know all of them. 
In real work, we just need to know a few of which are useful.

(1) KFold
Intro. 문제제기
데이터를 train, test, validation으로 split하는 것의 문제점?
전체 데이터를 훈련시키지 않게 됨. 즉, 훈련에 반영되지 않는 데이터 일부분이 존재하게 됨. (낭비되는 데이터)
하지만 그렇다고 모든 데이터를 훈련시키면 과적합 문제 발생.
또한 train으로 나누는 비율에 따라서 결과가 달라지는 문제 존재.

Body. cross validation (교차검증)의 필요성
Let's suppose, 전체 데이터를 5등분하여 각 구간을 test로 범위를 설정하여 5번 돌림.
**CAUTION** It may consume a whole lot of time for running the code.

In addtion, in order to use kfold (which is declared as a part of preprocessing), you need to replace fit and score parts with "cross_val_score".
So, with only this one line,
    scores = cross_val_score(model, x, y, cv=kfold)
you get to take care of fit, score, and kfold.
(It's where everybody gets to meet! Yay!)

To express this in a conceptual fashion, it could be summarized as follows:
    cross_val_score = fit + score + kfold
cf> m10_kfold_4_iris.py

In m11 files, you can check how to apply kfold (& cross_val_score) to all_estimators.
It will allow you to see all of the available algorithms for a given dataset.
cf> m11_kfold_estimators1_iris.py

https://nonmeyet.tistory.com/entry/KFold-Cross-Validation%EA%B5%90%EC%B0%A8%EA%B2%80%EC%A6%9D-%EC%A0%95%EC%9D%98-%EB%B0%8F-%EC%84%A4%EB%AA%85
https://devkor.tistory.com/entry/%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%9E%85%EB%AC%B8%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-%EC%84%A4%EB%AA%85-%EA%B5%90%EC%B0%A8-%EA%B2%80%EC%A6%9DK-Fold-Cross-Validation

(2) GridSearchCV, parameters, model.best_estimator_

"마, 인공지는한다 카는데 한땀한땀 하이퍼파라미터 수정하는 게 말이 되나!
그거 가내수공업 아이가? 이제 좀 산업혁명 쫌 해보자!"

순서 (cookbook style)
1. 임포트해준다 ( from sklearn.model_selection import GridSearchCV )
2. 파라미터 list of dict로 정해주자 ( parameters = [{"param1" : [value1, value2], "param2" : ["value1"]}, DICT, DICT ...] )
3. (화려한 조명 말고) model을 파라미터와 kfold와 함께 GridSearchCV로 감싸준다 ( model = GridSearchCV(SVC(), parameters, cv=kfold) )
4. 가장 좋은 모델을 알고 싶다면 model.best_estimator_ 를 사용하자
5. 분류냐 회귀냐에 따라서 accuracy_score 혹은 r2_score를 확인해보자

(3) RandomizedSearchCV

"어느 세월에 GridSearchCV를 이용해서 다 확인하나..."

GridSearchCV는 촘촘한 그물망처럼 모든 파라미터에 대해서 실행해서 **오지게** 오래 걸린다.
그래서 좀 다이어트 해서 일부 샘플만 확인하는 게 RandomizedSearchCV이다
DL의 Dropout과 좀 비슷하다.

할 일
1. GridSearchCV -> RandomizedSearchCV 로 바꿔준다 .끝.


내일은 전처리에 대해서 알아보도록 하자