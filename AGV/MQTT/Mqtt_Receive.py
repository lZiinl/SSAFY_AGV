import paho.mqtt.client as mqtt
import ipywidgets as widgets
from IPython.display import display
import cv2
import numpy as np

# MQTT 브로커 설정
broker_address = "127.0.0.1" 
topic = "agv0/image"

# 이미지 디스플레이 위젯 생성
image_widget = widgets.Image(format='jpeg', width=300, height=300)
display(image_widget)

# 수신한 메시지를 처리하는 콜백 함수
def on_message(client, userdata, message):
    # 메시지에서 이미지를 읽어들임
    img_data = np.frombuffer(message.payload, dtype=np.uint8)
    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    
    # 이미지를 JPEG 형식으로 변환하여 위젯에 표시
    _, jpeg = cv2.imencode('.jpg', frame)
    image_widget.value = jpeg.tobytes()

# MQTT 클라이언트 생성 및 설정
client = mqtt.Client()
client.on_message = on_message

# 브로커 연결 및 구독
client.connect(broker_address)
client.subscribe(topic)

# MQTT 클라이언트 시작
client.loop_start()