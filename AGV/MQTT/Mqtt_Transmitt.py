import traitlets
import ipywidgets
from IPython.display import display
from jetbot import Camera, bgr8_to_jpeg
import paho.mqtt.client as mqtt
import time
import cv2
import numpy as np

# MQTT 클라이언트 설정
broker_address = "127.0.0.1"
topic = "agv0/image"

client = mqtt.Client()
client.connect(broker_address)

# JetBot 카메라 설정
camera = Camera.instance(width=300, height=300)

# 카메라로 찍은 frame을 띄워줄 image 객체 생성, 크기를 맞출 필요는 없다.
image = ipywidgets.Image(format='jpeg', width=300, height=300)
display(image)

# MQTT 전송 및 이미지 처리 함수
def execute(camera_image):
    frame = np.copy(camera_image)
    if frame is not None:
        img_str = cv2.imencode('.jpg', frame)[1].tobytes()
        client.publish(topic, img_str)
    return bgr8_to_jpeg(frame)

# 위젯에 연결하여 실시간으로 보여주기
time.sleep(1)
traitlets.dlink((camera, 'value'), (image, 'value'), transform=execute)