import cv2 as cv

capture = cv.VideoCapture(0) #to open default camera

pretrained_model = cv.CascadeClassifier("face_dector.xml")

while True:
    boolean, frame = capture.read()   #returns 2 values, if image is captured then boolen is True
    if boolean == True: #image was captured
        gray = cv.cvtColor(frame, cv.COLOR_BG2GRAY) #converts the color to gray
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        # drawing rectangle in frame
        for (x,y,w,h) in coordinate_list:
            cv.rectangle(frame, (X,y), (x+w, y+h), (0,255,0), 2)

        #display detected face
        cv.imshow("Live Face Detection", frame)

        #condition to break out of while loop
        if cv.waitKey(20) == ord('x'):
            break

capture.release()
cv.destroyAllWindows()