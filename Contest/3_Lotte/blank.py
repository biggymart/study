import pandas as pd
import random

mine = pd.read_csv('C:/data/LPD_competition/lotte0324_1_copy.csv')
# print(mine)

empty_df = pd.DataFrame(columns=['filename', 'prediction'])
for i in range(72000):
    if i < 5832:
        pass
    else:
        rand_num = random.randint(0, 1000)
        empty_df = empty_df.append({'filename': '{0}.jpg'.format(i), 'prediction': rand_num}, ignore_index=True)
    
print(empty_df)
new_df = pd.concat([mine, empty_df])
print(new_df)
new_df.to_csv('C:/data/LPD_competition/lotte0323_temp.csv', index=False)
