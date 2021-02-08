# data
# OpenCV 
# http://pythonstudy.xyz/python/article/409-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%98%81%EC%83%81-%EC%B2%98%EB%A6%AC-OpenCV

### 1. OpenCV 다운로드
# pip install opencv-python


### 2. 이미지 파일 읽고 쓰기
import cv2

# 이미지 읽기
img = cv2.imread('test.jpg', 1)
 
# 이미지 화면에 표시
cv2.imshow('Test Image', img)
cv2.waitKey(0)
# 이미지 윈도우 삭제
cv2.destroyAllWindows()
 
# 이미지 다른 파일로 저장
cv2.imwrite('test2.png', img)


### 3. 카메라 영상 처리
import cv2
 
cap = cv2.VideoCapture(0)   # 0: default camera
#cap = cv2.VideoCapture("test.mp4") #동영상 파일에서 읽기
 
while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    if success:
        # 프레임 출력
        cv2.imshow('Camera Window', frame)
 
        # ESC를 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if (key == 27): 
            break
 
cap.release()
cv2.destroyAllWindows()


### 4. 카메라 영상 저장하기
import cv2
 
cap = cv2.VideoCapture(0); 
 
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("size: {0} x {1}".format(width, height))
 
# 영상 저장을 위한 VideoWriter 인스턴스 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test.avi', fourcc, 24, (int(width), int(height)))
 
while cap.isOpened():
    success, frame = cap.read()
    if success:
        writer.write(frame)  # 프레임 저장
        cv2.imshow('Video Window', frame)
 
        # q 를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else:
        break
 
cap.release()
writer.release()  # 저장 종료
cv2.destroyAllWindows()

### 5. OpenCV와 Matplotlib 활용
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('happyfish.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()