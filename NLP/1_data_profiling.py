# pip install -U pandas-profiling
import pandas as pd
import pandas_profiling
data = pd.read_csv('C:\data\csv\spam.csv',encoding='latin1')
# print(data[:5])
#      v1                                                 v2 Unnamed: 2 Unnamed: 3 Unnamed: 4
# 0   ham  Go until jurong point, crazy.. Available only ...        NaN        NaN        NaN
# 1   ham                      Ok lar... Joking wif u oni...        NaN        NaN        NaN
# 2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN        NaN        NaN
# 3   ham  U dun say so early hor... U c already then say...        NaN        NaN        NaN
# 4   ham  Nah I don't think he goes to usf, he lives aro...        NaN        NaN        NaN

pr = data.profile_report()
pr.to_file('./pr_report.html')