import csv
import random

# data = []  
# with open("C:/Users/ai/Downloads/LPD_competition/sample1.csv", "r") as the_file:
#     reader = csv.reader(the_file, delimiter=",")
#     for row in reader:
#         try:
#             line = str(random.randint(1,1000))
#             sid = line
#             new_row = [row[0], row[1],sid]
#             data.append(new_row)
#         except IndexError as error:
#             print(error)
#             pass
#     with open("Random.csv", "w+") as Ran_file:
#         writer = csv.writer(Ran_file, delimiter=",")
#         for new_row in data:
#             writer.writerow(new_row)

f = open('C:/Users/ai/Downloads/LPD_competition/sample.csv','r')
rdr = csv.reader(f)
lines = []
for line in rdr:
    if line[1] == 'prediction':
        line[1] = 'prediction'
        lines.append(line)
    else:
        line[1] = random.randint(1,1000)
        lines.append(line)
 
f = open('Random.csv','w',newline='') #원본을 훼손할 위험이 있으니 다른 파일에 저장하는 것을 추천합니다.
wr = csv.writer(f)
wr.writerows(lines)
 
f.close()