import warnings
warnings.filterwarnings('ignore')

#1. data
import numpy as np
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data 
y = dataset.target

#1-0. preprocessing
seed = 66
from sklearn.model_selection import train_test_split, KFold, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=seed)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) 

#2. model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

m1 = KNeighborsRegressor()
m2 = DecisionTreeRegressor()
m3 = RandomForestRegressor()
m4 = LinearRegression()
model_lst = [m1, m2, m3, m4]

#3. fit
#4. score
np.set_printoptions(precision=2)
for model in model_lst:
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print('{0} scores : {1}'.format(str(model), scores))

# KNeighborsRegressor() scores :    [0.37 0.35 0.32 0.52 0.41]
# DecisionTreeRegressor() scores :  [-0.12  0.09 -0.05  0.07 -0.11]
# RandomForestRegressor() scores :  [0.46 0.55 0.41 0.57 0.43]
# LinearRegression() scores :       [0.54 0.49 0.47 0.55 0.37]
