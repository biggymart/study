# cancer에 맞게 GridSearch 적용해서 만들어라
# pipeline ㄴㄴ, train_test_split ㅇㅇ

# f score
# xgboost parameters 알아보기, 많이 쓰는 것으로 제시해줌
# cross_val_score ㅇㅇ

parameters = [
    {"n_estimators" : [100, 200, 300], "learning_rate" : [0.001, 0.01, 0.1, 0.3], "max_depth" : [4, 5, 6]},
    {"n_estimators" : [90, 100, 110], "learning_rate" : [0.001, 0.01, 0.1], "max_depth" : [4, 5, 6], "colsample_bytree" : [0.6, 0.9, 1]},
    {"n_estimators" : [90, 110], "learning_rate" : [0.001, 0.1, 0.5], "max_depth": [4,5,6], "colsample_bytree" : [0.6, 0.9, 1], "colsample_bylevel" : [0.6, 0.7, 0.9]}
]
n_jobs = -1