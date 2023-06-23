import numpy as np
import cv2
import speech_recognition as sr

# sudo apt install espeak
# import pyaudio
# pip3 install pyttsx
# 3 sudo apt install espeak pip3 install pyaudio or use sudo apt install python3-pyaudio
# apt install libleptonica-dev tesseract-ocr  libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-deu tesseract-ocr-script-latn
import pyttsx3
import ultralytics
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.memory import ConversationBufferWindowMemory
from ultralytics import YOLO
import openai
import numpy as np
import time

time.sleep(5)
from transformers import pipeline

from test_langchain_com import listen

captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

API_KEY = ""

engine = pyttsx3.init()
voices = engine.getProperty('voices')
for idx, voice in enumerate(voices):
    print(idx, voice)
engine.setProperty('voice', voices[9].id)  # 9 = german

WIDTH = 1920
HEIGHT = 1080

GAPI_KEY = "AIzaSyAyz7m2buyH-Z-8iu5a7sDkkOoieShDMpE"

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ultralytics.checks()
# Create a new YOLO model from scratch
model = YOLO('../resources/yolov8.yaml')
model = YOLO('../resources/yolov8n.pt')


# for voice in engine.getProperty('voices'):
#    print(voice)


class DetectedObj:
    def __init__(self, name, pose, description):
        self.name = name
        self.pose = pose
        self.description = description
        self.obj_relationship = []

    def __str__(self):
        promp = f"{self.description}:{self.name};\n"  # xywhh:{self.pose};\n"
        promp = f"{self.name} "  # xywhh:{self.pose};\n"
        # if len(self.obj_relationship) > 0:
        #    promp += "Übschneidungen:"
        #    for relationship in self.obj_relationship:
        #        promp += str(relationship)
        #    promp += ";"
        return promp  # + "]"


def get_overlap(detectedObj1, detectedObj2):
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
            # return True
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
                # cv2.rectangle(frame, (int(result.boxes.xyxy[i, 0]), int(result.boxes.xyxy[i, 1])),
                #              (int(result.boxes.xyxy[i, 2]), int(result.boxes.xyxy[i, 3])), (255, 0, 0),
                #              thickness=1)
                # cv2.putText(frame, result.verbose().split(" ")[1].replace(",", ""),
                #            (int(result.boxes.xyxy[i, 0]), int(result.boxes.xyxy[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.9, (255, 0, 0),
                #
                detectedObjs.append(DetectedObj(result.verbose().split(" ")[1].replace(",", ""), (x, y, w, h),
                                                "Objekt"))

        except Exception as e:
            print(e)
            print("Unable to Recognize your image.")
            continue

    return detectedObjs


def bild_von_kamera():
    from PIL import Image
    _, frame = vid.read()
    # You may need to convert the color.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def screenshot_erstellen(width=WIDTH, height=HEIGHT, PIL=None):
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
            DetectedObj(text, (x, y, w, h), "Text"))

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


def describe_image(image):
    ret = captioner(image)
    return ret[0]["generated_text"]


img = screenshot_erstellen()
# img = bild_von_kamera()
text_detectedObjs = text_aus_bild_lesen(img)
yolo_detectedObjs = bilderkennung(img)

prompt = "image description: " + describe_image(img) + " text: "
for text in text_detectedObjs:
    # print(text)
    prompt += str(text)

prompt += " objects: "
for obj in yolo_detectedObjs:
    prompt += str(obj)

# overlap yolos?
# overlaps = []
# for yolo1 in yolo_detectedObjs:
#    for yolo2 in yolo_detectedObjs:
#        overlap = get_overlap(yolo1, yolo2)
#        if overlap != "":
#            overlaps.append(overlap)
#    for text in text_detectedObjs:
#        overlap = get_overlap(yolo1, text)
#        if overlap != "":
#            overlaps.append(overlap)
#
# prompt = ""
# for p in overlaps:
#    prompt += p + "\n"
# prompt = "texte: " + text_detectedObjs + " objekte: " + yolo_detectedObjs
# frage = speech2txt()

frage = listen()
# frage = speech2txt()
print("frage", frage)
print("prompt", prompt)
openai.api_key = API_KEY
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system",
#          "content": "Weiße nicht daraufhin, dass es schwierig ist. Es erübrigt sich darauf hinzuweisen, dass die Interpretation schwierig ist oder etwas ungenau ist. Bildbeschreiubung (Bildbeschreibung:) gefolgt von dem Text der Texterkennung (Text:) und Erkannte Objekte (Objekte:): %s." % prompt},
#         {"role": "user", "content": frage},
#     ]
# )

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.
Image description (image description): followed by the text of the text recognition (text:) and recognized objects (objects:): %s
{history}
Human: {human_input}
Assistant:"""%prompt

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0,openai_api_key=API_KEY),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Calibrating...")
    r.adjust_for_ambient_noise(source, duration=5)
    # optional parameters to adjust microphone sensitivity
    # r.energy_threshold = 200
    # r.pause_threshold=0.5

    print("Okay, go!")
    while 1:
        text = ""
        print("listening now...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=30)
            print("Recognizing...")
            # whisper model options are found here: https://github.com/openai/whisper#available-models-and-languages
            # other speech recognition models are also available.
            text = r.recognize_whisper(
                audio,
                model="base.en",
                show_dict=True,
            )["text"]
        except Exception as e:
            unrecognized_speech_text = (
                f"Sorry, I didn't catch that. Exception was: {e}s"
            )
            text = unrecognized_speech_text
        print(text)

        response_text = chatgpt_chain.predict(human_input=text)
        print(response_text)
        engine.say(response_text)
        engine.runAndWait()


result = ''
for choice in response.choices:
    result += choice.message.content
print(result)
speak(result)

# cv2.imshow('img', np.array(img))
# cv2.waitKey(0)
# print(speech2txt())
