import cv2
import time
import datetime

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

def run():
    detection = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 3
    SECONDS_TO_SCREENSHOT_FACE = 3

    frame_size = (int(cap.get(3)), int(cap.get(4))) 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")    

    while True:
        ret, frame = cap.read()

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame,"Press ESC to exit",(50,50), font, 1, (0, 255, 0),2, cv2.LINE_AA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, width, height) in faces: 
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

        if len(faces) + len (bodies) > 0:
            if detection:
                timer_started = False
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                faces = frame[y:y+height, x:x+width]
                cv2.imwrite('C:\Thesis\static\Datasets\Image'+ current_time + '.jpg', faces)
                cv2.rectangle(frame, (x,y),(x+width, y+height), (0,0,255),2) 
                out = cv2.VideoWriter( f"C:\Thesis\static\Recordings\{current_time}.mp4", fourcc, 20, frame_size) 
                print("Started Recording!")
        elif detection:
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:  
                    detection = False
                    timer_started = False
                    out.release()
                    print("Stop Recording!")
            else:
                timer_started = True
                detection_stopped_time = time.time()
        
        if detection:
            out.write(frame)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()