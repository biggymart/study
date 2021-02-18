import p71_byunsu as p71

print(p71.aaa)
print(p71.square(10))

print("================")

from p71_byunsu import aaa, square

aaa = 3
# 이 파일의 aaa와 p71_byunsu에 있는 aaa는 다른 메모리에 할당됨

print(p71.aaa)
print(p71.square(10))
