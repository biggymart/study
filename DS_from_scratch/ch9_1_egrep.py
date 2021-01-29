# 
import sys, re

print(sys.argv)
# ['c:\\Study\\DS_from_scratch\\ch9_1.py'], 커맨드라인에서 사용할 수 있는 모든 인자에 대한 list
# sys.argv[0]는 프로그램의 이름, sys.argv[1]는 커맨드라인에서 주어지는 정규표현식이다
regex = sys.argv[1]

for line in sys.stdin:
    if re.search(regex, line): # re.search(pattern: AnyStr, string: AnyStr)
        sys.stdout.write(line)