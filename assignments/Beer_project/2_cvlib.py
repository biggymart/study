# src: https://bskyvision.com/681

# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        break

    # apply object detection (물체 검출)
    bbox, label, conf = cv.detect_common_objects(frame)

    print(bbox, label, conf)

    # draw bounding box over detected objects (검출된 물체 가장자리에 바운딩 박스 그리기)
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    # display output
    cv2.imshow("Real-time object detection", out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()   


#######################

# import libraries
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

image_path = 'test1.JPG' # 여기에는 테스트할 이미지의 경로 및 이름을 넣어주시면 됩니다. 
im = cv2.imread(image_path) # 이미지 읽기


# object detection (물체 검출)
bbox, label, conf = cv.detect_common_objects(im)

print(bbox, label, conf)

im = draw_bbox(im, bbox, label, conf) 


cv2.imwrite('result.jpg', im) # 이미지 쓰기