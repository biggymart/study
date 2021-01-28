2021-01-27
range: m01 - m09

keywords: 
(1) brief history, 
(2) overall workflow of ML (+ comparison btwn DL & ML), 
(3) XOR problem, 
(4) 6 & 4 models (classification and regression), 
(5) sklearn.utils testing -> all_estimators

(1) Brief History
Although we began this course by learning DL, historically speaking, the concept of ML was introduced well before DL.
There are different views on when artificial intelligence actually took place. Some say it started from the ancient Greece; others pick different time. 
However, one of the most notable figure in the field of   
신경망은 엘런 튜닝 2차세계대전 즘에 (영화 이미테이션 게임) 소개됨. 튜링 머신. 그때부터 인공지능이 시작되었다는 사람도 있고 고대 그리스부터 시작되었다고 하는 사람이 있음. 아무튼 역사는 알아서 알아보고.
사과 이야기 했나? 

실질적으로 이 바닥이 발전한 것은 전쟁 때문이야. 상대편을 잡아내야하니까 암호해독이 필요. 튜링머신이라든지 여러가지 이야기가 나왔지만, 얘가 기계인가 아닌가 하는 튜링 게임.
방 안에 남자든 여자든 넣어두고 남자인지 여자인지 바깥에 있는 사람이 맞춰야 함. 글로 소통함. "뜨게질 좋아하냐?" 같은 질문을 통해 알아보게 됨. 방 안에 있는 사람은 최대한 맞추기 어렵게 하겠지?

그렇다면 이것을 응용한다면, 머신이 있어, 머신인지 사람인지 분별하는 게임도 생각해볼 수 있겠지.
챗봇이 약 70년 전부터 구현이 되었다는 거지. 뽀록이 나는 이유는 그 때 장비가 지금보다 안 좋아서임.
그런데 몇 년 전에 우크라이나인가 중국인의 방인가 해서 사람을 속임.

한 마디로 뭐야. 50년대 처음에 나왔을 때는 딥러닝 가능? 노노 신경망 개념은 있었지만 2000년대서야 제대로 발전함. 컴퓨터 사양 때문에. 
팬티엄에서 막 커졌잖아. 속도가 따라잡으니까... 알파고도 나오고. 그 전까지는 머신러닝의 세상이었음. 
캐글이라는 대회 5년 전만해도 텐서플로가 취급도 안 했음. XG부스터가 짱이었음.
속도가 빠르고, 장비의 재원이 먹히지 않겠지. GPU 없어도 됨. 

(2) Overall Workflow of ML (+ comparison btwn DL & ML)
# ML: data -> model -> fit -> score and predict
# (1) 전처리에서 OneHotEncoding 불필요 
# (2) 모델에서 히든레이어 구성 불필요
# (3) 컴파일 불필요 
# (4) evaluate 대신 score 사용, evaluate은 loss와 metrics를 반환하는 한편 score는 회귀/분류에 맞는 지표를 반환
cf> m02_ml_sample.py

(3) XOR problem, 
XOR 문제 해결 방식 2가지
(1) ML: LinearSVC -> SVC  (cf> m03_4_xor2.py)
(2) DL: hidden layer 구성 및 activation 'sigmoid' -> 'relu'  (cf> m03_6_xor_keras2.py)

(4) 6 & 4 models (classification and regression), 
# sklearn이 keras보다 먼저 나온 모델임
# classification by ML (6가지)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
# regression by ML (4가지)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

### (유치한) 암 기 법 ###
Classifier or Regressor
"나무" 주위에서     --> DecisionTree
"이웃"들이          --> KNeighbors
"일직선"으로 서서    --> Logistic/Linear + Regression
"앙상블" 연주를 한다 --> RandomForest

(5) sklearn.utils testing -> all_estimators
# 선생님은 m09의 파일명을 selectModel 이라고 명명했지만 (모든 모델을 쫙 깔아놓고 골라라골라 하는 거라는 의미에서)
# 나는 원문에 더 가깝게 하기 위해서 allEstimators 라고 했다


