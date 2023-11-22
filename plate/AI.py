from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt
import os

# font 
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

def ocr_image_with_paddle(img, coordinates):
    x,y,w,h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])
    img = img[y:h,x:w]
    reader = PaddleOCR(show_log=False, use_angle_cls=True, drop_score=0.5)
    results = reader.ocr(img)
    if results == [[]]:
        return "", 0

    plate_chars = ''
    conf = 0
    for idx in range(len(results)):
        text, conf = results[idx][1]
        plate_chars = plate_chars + text

    return str(plate_chars), conf

def detection_(img):
    # model_path = os.getcwd() + "\\plate_prediction.pt"
    model_path = "C:/Users/vanhu/.spyder-py3/plate/plate_prediction.pt"
    lisense_plate_detector = YOLO(model_path)
    # per = lisense_plate_detector.predict(show=True, source = img_path)
    results = lisense_plate_detector(img)[0]
    detections = []
    for detection in results.boxes.data.tolist():
        # x1, y1, x2, y2, score, class_id = detection
        # print(detection)
        detections.append(detection)
    for predict in detections:
        zone = [predict[0], predict[1], predict[2], predict[3]]
        x1, y1, x2, y2 = int(predict[0]), int(predict[1]), int(predict[2]), int(predict[3])
        plate_char, conf = ocr_image_with_paddle(img, zone)
        print(plate_char)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, plate_char,  (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        plt.imshow(img)
    return img
    
def main():
    img_path = "C:/Users/vanhu/.spyder-py3/plate/test1.jpg"
    # img_path = os.getcwd() + "\\test1.jpg"
    img = cv2.imread(img_path)
    detection_(img)

if __name__ == "__main__":
    main()