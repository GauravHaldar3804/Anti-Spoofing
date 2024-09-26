from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone

offsetWratio = 10
offsetHratio = 10
    # Initialize the webcam
    # '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
cap = cv2.VideoCapture(0)

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

        # Check if any face is detected
    if bboxs:
            # Loop through each bounding box
        for bbox in bboxs:
                # bbox contains 'id', 'bbox', 'score', 'center'

                # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

                # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(img,(x ,y, w, h),(255,0,255),3)

            offsetW = (offsetWratio/100)*w

            x = int(x - offsetW)
            w = int(w + offsetW *2)

            offsetH = (offsetHratio/100)*h

            y = int(y - offsetH*4)
            h = int(h + offsetH *4.5)

            cv2.rectangle(img,(x ,y, w, h),(255,0,0),3)



        # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
        # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)