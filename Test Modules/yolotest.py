from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import torch


# Check for GPU availability
if torch.cuda.is_available():
    print("YOLOv8 is running on GPU!")
else:
    print("YOLOv8 is running on CPU.")

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
# cap = cv.VideoCapture("Chapter 6 Webcam YOLO/Videos/cars.mp4")




classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("Models/yolov8n.pt")

while True:
    succes , img = cap.read()
    img = cv.flip(img,1)
    results = model(img,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1 , y1 , x2 , y2 = box.xyxy[0]
            x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2) 
            cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)

            #Confidence
            conf = math.ceil(box.conf[0]*100) / 100
            # Class Names
            cls = int(box.cls[0])
            cvzone.putTextRect(img,f"{classNames[cls]} {conf}",(max(0,x1),max(35,y1)),scale=1,thickness=2,offset=3)
    cv.imshow("Webcam",img)
    cv.waitKey(1)

