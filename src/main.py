
import numpy as np
import cv2
import speech_recognition as sr
import pyttsx3
import ultralytics
from ultralytics import YOLO
import openai
import pyaudio
#p = pyaudio.PyAudio()
#for i in range(p.get_device_count()):
#    print (p.get_device_info_by_index(i))
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
# change_voice(engine, "nl_BE", "VoiceGenderFemale")
WIDTH = 2560
HEIGHT = 1440


vid = cv2.VideoCapture(1, cv2.CAP_DSHOW) # this is the magic!
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ultralytics.checks()
# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')#


for voice in engine.getProperty('voices'):
    print(voice)


class DetectedObj:
    def __init__(self, name, pose, description):
        self.name = name
        self.pose = pose
        self.description = description
        self.obj_relationship = []
    def __str__(self):
        promp = f"Objecktsignatur[Beschreibung:{self.description};Bezeichnung:{self.name};xywhh:{self.pose};"
        if len(self.obj_relationship) > 0:
            promp += "Übschneidungen:"
            for relationship in self.obj_relationship:
                promp += str(relationship)
            promp += ";"
        return promp + "]"

def get_overlap(detectedObj1,  detectedObj2):
    name1 = detectedObj1
    x1, y1, w1, h1 = detectedObj1.pose
    name2 = detectedObj2
    x2, y2, w2, h2 = detectedObj2.pose

    # Überprüfung der Überlappung in horizontaler Richtung
    if x1 < x2 + w2 and x1 + w1 > x2:
        # Überlappung in horizontaler Richtung
        # Überprüfung der Überlappung in vertikaler Richtung
        if y1 < y2 + h2 and y1 + h1 > y2:
            return f"{name1}, {name2} Überlappung in vertikaler Richtung"
            #return True
        elif y1 >= y2 and y1 + h1 <= y2 + h2:
            return f"{name1} liegt vollständig innerhalb von {name2} in vertikaler Richtung"
        elif y2 >= y1 and y2 + h2 <= y1 + h1:
            return f"{name2} liegt vollständig innerhalb von {name1} in vertikaler Richtung"
    elif x1 >= x2 and x1 + w1 <= x2 + w2:
        # rect1 liegt vollständig innerhalb von rect2 in horizontaler Richtung
        # Überprüfung der Überlappung in vertikaler Richtung
        if y1 < y2 + h2 and y1 + h1 > y2:
            return f"{name1}, {name2} Überlappung in vertikaler Richtung"
        elif y1 >= y2 and y1 + h1 <= y2 + h2:
            return f"{name1} liegt vollständig innerhalb von {name2} in vertikaler Richtung"
        elif y2 >= y1 and y2 + h2 <= y1 + h1:
            return f"{name2} liegt vollständig innerhalb von {name1} in vertikaler Richtung"

    # Keine Überlappung oder Einschluss
    return ""

def prompt_erkennung(obj_pose_tuble, text_pose_tupel):
    # check_overlap
    prompt = {}
    for text, (tx, ty, tw, th) in text_pose_tupel:
        is_text_on_obj = False
        for obj, (ox, oy, ow, oh) in obj_pose_tuble:
            pass





def bilderkennung(frame):
    frame = np.array(frame)
    results = model(frame)
    # Bounding Boxes zeichnen
    for result in results:
        # Detection
        detectedObjs = []
        try:
            for i in range(result.boxes.xywh.shape[0]):
                (x, y, w, h) = result.boxes.xywh[i]
                (x, y, w, h) = (int(x), int(y), int(w), int(h))
                #cv2.rectangle(frame, (int(result.boxes.xyxy[i, 0]), int(result.boxes.xyxy[i, 1])),
                #              (int(result.boxes.xyxy[i, 2]), int(result.boxes.xyxy[i, 3])), (255, 0, 0),
                #              thickness=1)
                #cv2.putText(frame, result.verbose().split(" ")[1].replace(",", ""),
                #            (int(result.boxes.xyxy[i, 0]), int(result.boxes.xyxy[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.9, (255, 0, 0),
                #
                detectedObjs.append(DetectedObj(result.verbose().split(" ")[1].replace(",", ""), (x, y, w, h),
                                                "Objekt mit Yolov8 erkannt."))


        except Exception as e:
            print(e)
            print("Unable to Recognize your image.")
            continue

    return detectedObjs

def bild_von_kamera():
    _, frame = vid.read()
    return frame


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def screenshot_erstellen(width=WIDTH, height=HEIGHT):
    from PIL import ImageGrab
    img = ImageGrab.grab(bbox=(0, 0, width, height))
    return img


def text_aus_bild_lesen(frame):
    import pytesseract
    from pytesseract import Output
    d = pytesseract.image_to_data(frame, output_type=Output.DICT)
    n_boxes = len(d['level'])
    detectedObjs = []
    for i in range(n_boxes):
        (x, y, w, h) = (int(d['left'][i]), int(d['top'][i]), int(d['width'][i]), int(d['height'][i]))
        text = d['text'][i]
        if text.replace(" ", "") == "":
            continue
        detectedObjs.append(
            DetectedObj(text, (x, y, w, h), "Text mit pytesseract erkannt."))

    return detectedObjs

def speech2txt():
    r = sr.Recognizer()
    sr.Microphone()
    while True:
        with sr.Microphone() as source:

            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language='de-in')
            print(f"User said: {query}\n")
            return query

        except Exception as e:
            print(e)
            print("Unable to Recognize your voice.")
            continue

    return query

#speak("HALLO wie geht es dir`?")

img = screenshot_erstellen()
text_detectedObjs = text_aus_bild_lesen(img)
yolo_detectedObjs = bilderkennung(img)

# overlap yolos?
overlaps = []
for yolo1 in yolo_detectedObjs:
    for yolo2 in yolo_detectedObjs:
        overlap = get_overlap(yolo1, yolo2)
        if overlap != "":
            overlaps.append(overlap)
    for text in text_detectedObjs:
        overlap = get_overlap(yolo1, text)
        if overlap != "":
            overlaps.append(overlap)

prompt = ""
for p in overlaps:
    prompt += p + "\n"

while True:
    key_btn = {"Landeanfrage":"n"}
    frage = speech2txt()

    openai.api_key = "sk-"
    print(text)
    # print(openai.Model.list())
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Dein Name ist Sarah. Reagiere nur wenn du angesprochen wirst. Interpretion von einem VoiceCommand ob eine Taste gedrückt werden soll oder nicht. Handelt es sich dabei um eine Action welche in dem Dict (Action, Taste) mit dem Inhalt" +str(key_btn) + " enthalten ist? Falls ja gib nur die entsprechende Taste als Antwort zurück, ansonsten False und nichts anderes." },
            {"role": "user", "content": frage},
        ]
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content
    print(result)
    speak(result)


import cv2
import numpy as np
cv2.imshow('img', np.array(img))
cv2.waitKey(0)
#print(speech2txt())
import sys
sys.exit()

import ultralytics
from ultralytics import YOLO
# import the opencv library
import cv2

import cv2
import numpy as np
from PIL import ImageGrab
from time import sleep
import pytesseract
import time
from pygame import mixer
from copy import deepcopy
from imageai.Detection import ObjectDetection
import os
import torch
# define a video capture object
vid = cv2.VideoCapture(2)
vid = cv2.VideoCapture(1, cv2.CAP_DSHOW) # this is the magic!

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ultralytics.checks()
# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')#

mixer.init()
audio_playing = False
think = "Folgendes ist bereits bekannt: "


while (True):


    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    results = model(frame)

    # Bounding Boxes zeichnen
    for result in results:
        # Detection
        print(result.names)
        result.boxes.xyxy  # box with xyxy format, (N, 4)
        result.boxes.xywh  # box with xywh format, (N, 4)
        result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        result.boxes.conf  # confidence score, (N, 1)
        result.boxes.cls  # cls, (N, 1)

        # Segmentation
        #result.masks.data  # masks, (N, H, W)
        #result.masks.xy  # x,y segments (pixels), List[segment] * N
        #result.masks.xyn  # x,y segments (normalized), List[segment] * N

        # Classification
          # cls prob, (num_class, )
        try:
            cv2.rectangle(frame, (int(result.boxes.xyxy[0,0]), int(result.boxes.xyxy[0,1])), (int(result.boxes.xyxy[0,2]), int(result.boxes.xyxy[0,3])), (255, 0, 0),
                          thickness=1)
            cv2.putText(frame, result.verbose().split(" ")[1].replace(",",""), (int(result.boxes.xyxy[0,0]), int(result.boxes.xyxy[0,1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                        1)
        except:
            pass
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    openai.api_key = "sk-"
    print(text)
    # print(openai.Model.list())
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Eine Objekterkennung von einem Bild gibt dir Informationen über die Umgebenung. Du gibst diese weiter zu einem Blindenmenschen."},
            {"role": "user", "content": "Was ist hier zu sehen?  " +  result.verbose().split(" ")[1].replace(",","")},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    #think += deepcopy(result)
    #print(think)
    from gtts import gTTS

    while mixer.music.get_busy():  # wait for music to finish playing
        time.sleep(1)

    mixer.music.unload()

    language = 'de'
    tts = gTTS(text=result,
               lang=language,
               slow=False)
    tts.save("tts.mp3")

    mixer.music.load('tts.mp3')
    mixer.music.play()

mixer.quit()

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



