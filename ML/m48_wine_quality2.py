import pandas as pd
import numpy as np


wine = pd.read_csv('../data/csv/winequality-white.csv',
    sep=';', header=0, index_col=None
)

count_data = wine.groupby('quality')['quality'].count()
print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

# print(np.unique(count_data['quality']))

import matplotlib.pyplot as plt
count_data.plot()
plt.show()
# 꽤나 몰려있으니까 카테고리를 줄여보는 게 어떨까




