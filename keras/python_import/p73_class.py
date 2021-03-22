class Person:
    def __init__(self, name, age, address): # self는 클래스 자체를 말함, 필수요소임
        self.name = name #1. 변수 넣을 수 있고
        self.age = age
        self.address = address
    
    def greeting(self): #2. 함수도 넣을 수 있고
        print("안녕하세요, 저는 {0}입니다.".format(self.name))
