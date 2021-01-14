import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)

print(df)
# index_col 파라미터 없이는 뭔가 조금 이상하지 않니?
# index가 데이터로 들어감, index가 데이터가 아님을 명시해줘야 함.

# header가 있다고 디폴트로 여기기 때문에 (header=0)
# header 없는 경우에는 header=None으로 설정해주어야 함