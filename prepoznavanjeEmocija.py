import cv2
from deepface import DeepFace
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video.isOpened():
    raise IOError("Cannot open webcam")

while video.isOpened():
    _, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x, y, w, h in face:
        try:
            image = cv2.rectangle(frame, (x, y), (x + w, y + h), (199,21,133), 1)
            analyze = DeepFace.analyze(frame, actions=['emotion'])

            if analyze[0]['dominant_emotion'] == 'angry':
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (12, 12, 232), 2)

            elif analyze[0]['dominant_emotion'] =='disgust':
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 235, 79), 2)

            elif analyze[0]['dominant_emotion'] == 'fear':
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 52, 214), 2)

            elif analyze[0]['dominant_emotion'] == 'happy':
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 192, 235), 2)

            elif analyze[0]['dominant_emotion'] == 'sad':
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 95, 52), 2)

            elif analyze[0]['dominant_emotion'] == 'surprise':
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (215, 163, 240), 2)

            else:
                cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 111, 114), 2)
            print(analyze[0]['dominant_emotion'])
        except:
            print("No face")

    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()