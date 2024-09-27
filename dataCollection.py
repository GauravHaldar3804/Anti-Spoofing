from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time
############################
imageOutputPath = "Datasets/Data Collect"
offsetWratio = 10
offsetHratio = 20
confidence = 80
debug = False
save = True
floatingpoint = 6
blurThreshold = 35
#############################
camWidth,camHeight = 640,480
    # Initialize the webcam
    # '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)

    # Initialize the FaceDetector object
    # minDetectionCon: Minimum detection confidence threshold
    # modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
detector = FaceDetector()

    # Run the loop to continually get frames from the webcam
while True:
        # Read the current frame from the webcam
        # success: Boolean, whether the frame was successfully grabbed
        # img: the captured frame
    success, img = cap.read()

        # Detect faces in the image
        # img: Updated image
        # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)
    imgOUT = img.copy()

        # Check if any face is detected
    if bboxs:
            # Loop through each bounding box
        listBlur = [] # Adding value of faces true if detected and false if not
        listInfo = [] # Contains normalised values and classnames
        for bbox in bboxs:
                # bbox contains 'id', 'bbox', 'score', 'center'

                # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            

            # ---- Checking Confidence ---- #

            if score > confidence:

                # ---- Adding offset to found bounding box ---- #
                offsetW = (offsetWratio/100)*w

                x = int(x - offsetW)
                w = int(w + offsetW *2)

                offsetH = (offsetHratio/100)*h

                y = int(y - offsetH*3)
                h = int(h + offsetH *3.5)

                # ---- Avoiding values below 0 ---- #

                if x < 0 : x = 0
                if y < 0 : y = 0
                if w < 0 : w = 0
                if h < 0 : h = 0


                # ---- Finding Bluriness ---- #
                
                imgFace = img[y:y+h,x:x+w]
                blurValue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())

                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)


                # ---- Normalization ---- #
                iw , ih,_ = img.shape

                xc = x + w/2
                yc = y + h/2

                xcn = round(xc/iw,floatingpoint)
                ycn = round(yc/ih,floatingpoint)
                wn = round(w/iw,floatingpoint)
                hn = round(h/ih,floatingpoint)
                

                # ---- Avoiding values above 1 ---- #

                if xcn > 1 : xcn = 1
                if ycn > 1 : ycn = 1
                if wn > 1 : wn = 1
                if hn > 1 : hn = 1
                # print(xcn,ycn,wn,hn)


                # ---- Drawing ---- #
                
                cv2.imshow("Face",imgFace)
                cv2.rectangle(imgOUT,(x ,y, w, h),(255,0,0),3)
                cvzone.putTextRect(imgOUT,f"Score:{score}% Blur:{blurValue}",(x,y+20),scale=2,thickness=3)
                if debug == True:
                    cv2.imshow("Face",imgFace)
                    cv2.rectangle(img,(x ,y, w, h),(255,0,0),3)
                    cvzone.putTextRect(img,f"Score:{score}% Blur:{blurValue}",(x,y+20),scale=2,thickness=3)


        if save == True:

            if all(listBlur) and listBlur != []:

                timeNow = time()
                timeNow = str(timeNow).split(".")
                timeNow = int(timeNow[0]+timeNow[1])

                cv2.imwrite(f"{imageOutputPath}/{timeNow}.jpg",img)

                print(timeNow)
            
    



        # Display the image in a window named 'Image'
    cv2.imshow("Image", imgOUT)
        # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)