import p11_car
import p12_tv

# 외부에서 import해서 __name__을 출력하면 __main__이 아니라 p11_car처럼 파일 이름이 출력됨

print("==========")
print("p13_do.py의 module 이름은: ", __name__)
print("==========")

p11_car.drive()
p12_tv.watch()