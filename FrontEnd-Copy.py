import tkinter as tk
import cv2 
from tkinter import Message, Text
from tkinter import*
import os
import csv
import numpy as np 
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
from time import strftime
import tkinter.ttk as ttk
import tkinter.font as font
from pathlib import Path
from tkinter import filedialog
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from keras.preprocessing import image 
from keras.utils import img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
#from Feature_Extractor import FeatureExtractor
from test2 import run
#from offline import train
#from Counter import counter



cap= cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16 (weights='imagenet')
        self.model = Model (inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        pass
        
    def extract(self, img):
        img = img.resize((224,224)).convert("RGB") 
        x = image.img_to_array(img)
        x = np.expand_dims (x, axis=0)
        x = preprocess_input(x)
        feature= self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    
window = tk.Tk()

window.title("Human_Detection_System")
window.geometry("900x600")
window.resizable(width=False, height=False) 
window.configure(background="white")

bg = PhotoImage(file="background.png")

my_label = Label (window, image=bg)
my_label.place(x=0,y=0,relwidth=1,relheight=1)

message= tk.Label(
    window, text ="Human Detection System", bg="gray",
    fg="black", width= 20,
    height=3, font=('Times', 15, 'bold'))
message.place(x=10, y=0)
def train():
    if __name__=="__main_":

            fe = FeatureExtractor()

    for img_path in sorted(Path("C:\Thesis\static\Datasets").glob ("*.jpg")): 
            print(img_path)
            fe = FeatureExtractor()

            Feature=fe.extract(img=Image.open(img_path))

            feature_path = Path("C:\Thesis\static\Feature") / (img_path.stem + ".npy") 
            print(feature_path)

            np.save(feature_path, Feature)
def opensearchwindow():
    searchwindow = Toplevel()
    searchwindow.title("Search Window")
    searchwindow.geometry ("200x250")
    searchwindow.resizable (width=False, height=False)
    searchwindow.configure(bg="black")

    img = Image.open("C:\\Thesis\\searchwindowbg.png") 
    resized_image = img.resize((100, 100), Image.Resampling.LANCZOS) 
    converted_image = ImageTk. PhotoImage(resized_image)

    image_label = tk.Label(searchwindow, image = converted_image, width=100, height=100) 
    image_label.place(x=50, y=30)

    train_btn = tk.Button(searchwindow, bg="gray", text="Train Dataset", font=10, width= 15, height =1,activebackground="red", command=train)
    train_btn.place(x=28, y=150)
    searchimage_btn = tk.Button(searchwindow, bg="gray", text="Search Dataset", font=10, width= 15, height =1, command=search_engine) 
    searchimage_btn.place(x=28, y=190)
    searchwindow.mainloop()
def search_engine():
    app = Flask(__name__)

    # Read image features
    fe = FeatureExtractor()
    features = []
    img_paths = []
    for feature_path in Path("./static/feature").glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/Datasets") / (feature_path.stem + ".jpg"))
    features = np.array(features)


    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            file = request.files['query_img']

            # Save query image
            img = Image.open(file.stream)  # PIL image
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
            img.save(uploaded_img_path)

            # Run search
            query = fe.extract(img)
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            ids = np.argsort(dists)[:30]  # Top 30 results
            scores = [(dists[id], img_paths[id]) for id in ids]

            return render_template('index.html',
                                query_path=uploaded_img_path,
                                scores=scores)
        else:
            return render_template('index.html')


    if __name__=="__main__":
        app.run("0.0.0.0")
def open_dataset():
    return filedialog.askopenfilename (initialdir="C:\Thesis\static\Datasets")
def open_recordings():
    return filedialog.askopenfilename (initialdir="C:\Thesis\static\Recordings") 
def open_uploads():
    return filedialog.askopenfilename (initialdir=r"C:\Thesis\static\Uploaded")
def counter():
    root = tk.Tk()
    root.withdraw()
    #reference_file_name = filedialog.askopenfilename()
    file_name = filedialog.askopenfilename()
    cap = cv2.VideoCapture(file_name)
    # fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=150)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=150, backgroundRatio=0.3)

    def line1(x,y):
        return y - (29*x)/96.0 - 300

    def line2(x,y):
        return y - (29*x)/96.0 - 500

    crossedAbove = 0
    crossedBelow = 0
    points = set()
    pointFromAbove = set()
    pointFromBelow = set()
    H = 1080
    W = 1980
    OffsetRefLines = 50  #Adjust ths value according to your usage


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_output.avi',fourcc, 20.0, (W, H))
    font = cv2.FONT_HERSHEY_SIMPLEX
    while(1):
        pointInMiddle = set()
        prev = points
        points = set()
        ret, frame1 = cap.read()
        frame = cv2.resize(frame1,(W, H))
        height = np.size(frame,0)
        width = np.size(frame,1)
        #print(height)
        #print(width)
        fgmask = frame
        fgmask = cv2.blur(frame, (10,10))
        fgmask = fgbg.apply(fgmask)
        fgmask = cv2.medianBlur(fgmask, 7)
        oldFgmask = fgmask.copy()
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, 1)
        for contour in contours:
            if  2000 <= cv2.contourArea(contour) <= 200000:
                #QttyOfContours = QttyOfContours+1   
                x,y,w,h = cv2.boundingRect(contour)
                if 30<w<500 and 70<h<700:
                    cv2.drawContours(frame, contour, -1, (0, 0, 255), 2)
                    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2, lineType=cv2.LINE_AA)
                    point = (int(x+w/2.0), int(y+h/2.0))
                    points.add(point)
        for point in points:
            (xnew, ynew) = point
            # if line1(xnew, ynew) > 0 and line2(xnew, ynew) < 0:
            #     pointInMiddle.add(point)
            for prevPoint in prev:
                (xold, yold) = prevPoint
                dist = cv2.sqrt((xnew-xold)*(xnew-xold)+(ynew-yold)*(ynew-yold))
                if dist[0] <= 120:
                    if line1(xnew, ynew) >= 0 and line2(xnew, ynew) <= 0:
                        if line1(xold, yold) < 0: # Point entered from line above
                            pointFromAbove.add(point)
                        elif line2(xold, yold) > 0: # Point entered from line below
                            pointFromBelow.add(point)
                        else:   # Point was inside the block
                            if prevPoint in pointFromBelow:
                                pointFromBelow.remove(prevPoint)
                                pointFromBelow.add(point)

                            elif prevPoint in pointFromAbove:
                                pointFromAbove.remove(prevPoint)
                                pointFromAbove.add(point)

                    if line1(xnew, ynew) < 0 and prevPoint in pointFromBelow: # Point is above the line
                        print('One Crossed Above')
                        print(point)
                        crossedAbove += 1
                        pointFromBelow.remove(prevPoint)

                    if line2(xnew, ynew) > 0 and prevPoint in pointFromAbove: # Point is below the line
                        print('One Crossed Below')
                        print(point)
                        crossedBelow += 1
                        pointFromAbove.remove(prevPoint)


        for point in points:
            if point in pointFromBelow:
                cv2.circle(frame, point, 3, (255,0,255),6)
            elif point in pointFromAbove:
                cv2.circle(frame, point, 3, (0,255,255),6)
            else:
                cv2.circle(frame, point, 3, (0,0,255),6)

        #plot reference lines (entrance and exit lines) 
        # CoorYEntranceLine = int((height / 2)-OffsetRefLines)
        # CoorYExitLine = int((height / 2)+OffsetRefLines)
        # #print(CoorYEntranceLine)
        # #print(CoorYExitLine)
        # cv2.line(frame, (0,CoorYEntranceLine), (width,CoorYEntranceLine), (255, 0, 0), 4)
        # cv2.line(frame, (0,CoorYExitLine), (width,CoorYExitLine), (0, 0, 255), 4)

        cv2.line(frame, (0,300), (width,height-200), (255, 0, 0), 4)
        cv2.line(frame, (0,500), (width,height), (255, 0, 0), 4)


        # cv2.line(frame, (0,300), (1920,880), (255, 0, 0), 4)
        # cv2.line(frame, (0,500), (1920,1080), (255, 0, 0), 4)
        cv2.putText(frame,'Exit = '+str(crossedAbove),(100,50), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,'Entry = '+str(crossedBelow),(100,100), font, 1,(0,0,0),2,cv2.LINE_AA)
        
        
        cv2.imshow('frame',frame)
        out.write(frame)
        l = cv2.waitKey(1) & 0xff
        if l == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    detection = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 3
    SECONDS_TO_SCREENSHOT_FACE = 3

    frame_size = (int(cap.get(3)), int(cap.get(4))) 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")    

    while True:
        ret, frame = cap.read()

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

search_btn = tk.Button(window, bg="gray", text= "Search", font=10, width= 15, height =1, command=opensearchwindow)
search_btn.place(x=690, y=60)
counter_btn = tk.Button(window, bg="gray", text= "Counter", font=10, width= 15, height =1, command = counter )
counter_btn.place(x=690, y=160)
dataset_btn = tk.Button(window, bg="gray", text= "Dataset", font=10, width= 15, height =1, command=open_dataset)
dataset_btn.place(x=690, y=260)
recordings_btn = tk. Button (window, bg="gray", text= "Recordings", font=10, width= 15, height =1, command=open_recordings)
recordings_btn.place(x=690, y=360)
uploads_btn = tk.Button(window, bg="gray", text= "Past Searches", font=10, width= 15, height =1, command=open_uploads) 
uploads_btn.place(x=690, y=460)
record_btn = tk.Button(window, bg="gray", text= "Record", font=10, width= 15, height =1, command=run)
record_btn.place(x=20, y=550)

   

label = tk.Label(window)

label.place(x=20, y=58) # to resize label when resize window

def show_frame():

    ret, frame = cap.read()

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame,strftime("%H: %M: %S"),(50,50), font, 1, (0, 255, 0),2, cv2.LINE_AA)

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        bodies = face_cascade.detectMultiScale (gray, 1.3, 5)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    # convert to Tkinter image 
    photo = ImageTk. PhotoImage(image=img)

    # solution for bug in PhotoImage

    label.photo = photo

    # replace image in label 
    label.configure(image=photo) 
    window.after (20, show_frame)

    # start function which shows frame
show_frame()

window.mainloop()

cap.release()


