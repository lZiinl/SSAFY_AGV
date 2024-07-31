#최종 코드

from sense_hat import SenseHat
import time
from gpiozero import DistanceSensor
import paho.mqtt.client as mqtt
from gpiozero import DigitalInputDevice
from firebase import firebase
from datetime import datetime
import pytz
import time
import json

# GPIO 핀 번호 설정
sensor1_trigger = 27
sensor1_echo = 17
sensor2_trigger = 15
sensor2_echo = 14
sensor3_trigger = 21
sensor3_echo = 20
LIGHT_SENSOR_PIN = 16

sense = SenseHat()

# 초음파 센서 객체 생성
sensor1 = DistanceSensor(echo=sensor1_echo, trigger=sensor1_trigger)
sensor2 = DistanceSensor(echo=sensor2_echo, trigger=sensor2_trigger)
sensor3 = DistanceSensor(echo=sensor3_echo, trigger=sensor3_trigger)

# 조도 센서 초기화
light_sensor = DigitalInputDevice(LIGHT_SENSOR_PIN)

# LED brightness
red = (155, 0, 0)
green = (0, 155, 0)
white = (155, 155, 155)
out = (0, 0, 0)
angle = 0

# Firebase Realtime Database에 연결하는 Firebase 객체 생성
# project host url
firebase_url = ""
firebase = firebase.FirebaseApplication(firebase_url, None)

# 한국 시간대 (Asia/Seoul)로 설정
korea_timezone = pytz.timezone("Asia/Seoul")

# MQTT 설정
mqtt_broker = "192.168.110.120"  # 예: "localhost" 또는 브로커의 올바른 호스트 이름/주소
mqtt_port = 1883
mqtt_topic1 = "sensor/data1"
mqtt_topic2 = "sensor/data2"
mqtt_topic_heading = "sensor/data/heading"
mqtt_topic_direction = "sensor/data/direction"
mqtt_topic_distance1 = "sensor/data/distance1"
mqtt_topic_distance2 = "sensor/data/distance2"
mqtt_topic_distance3 = "sensor/data/distance3"
mqtt_topic_humidity = "sensor/data/humidity"
mqtt_topic_temperature = "sensor/data/temperature"
mqtt_topic_led1 = "sensor/data/led1"
mqtt_topic_led2 = "sensor/data/led2"
mqtt_topic_led3 = "sensor/data/led3"
mqtt_topic_light = "sensor/data/light"
mqtt_topic_sig = "agv/sig"

# MQTT 클라이언트 초기화
client = mqtt.Client()

def get_heading():
    # Get the orientation data
    orientation = sense.get_orientation_degrees()
    pitch = orientation['pitch']
    roll = orientation['roll']
    yaw = orientation['yaw']

    # Print the yaw angle
    #print(f"Pitch: {pitch:.2f} Roll: {roll:.2f} Yaw: {yaw:.2f}")
    return yaw

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

client.on_connect = on_connect
client.connect(mqtt_broker, mqtt_port, 60)

try:
    while True:
        # 거리 측정
        distance1 = sensor1.distance * 100  # 거리를 센티미터로 변환
        distance2 = sensor2.distance * 100
        distance3 = sensor3.distance * 100

        humid = sense.get_humidity()
        temp = sense.get_temperature()
        sensor_value = light_sensor.value
        sig=""

        heading = get_heading()
        if heading < 22.5 or heading >= 337.5:
            direction = "North"
        elif 22.5 <= heading < 67.5:
            direction = "Northeast"
        elif 67.5 <= heading < 112.5:
            direction = "East"
        elif 112.5 <= heading < 157.5:
            direction = "Southeast"
        elif 157.5 <= heading < 202.5:
            direction = "South"
        elif 202.5 <= heading < 247.5:
            direction = "Southwest"
        elif 247.5 <= heading < 292.5:
            direction = "West"
        elif 292.5 <= heading < 337.5:
            direction = "Northwest"

        if distance3 <= 10 or distance2 <= 10 or distance1 <= 10:
            client.publish(mqtt_topic_led1,"false")
            client.publish(mqtt_topic_led2,"false")
            client.publish(mqtt_topic_led3,"false")
            client.publish(mqtt_topic_sig,"stop")

            for i in range(8):
                for j in range(8):
                    sense.set_pixel(i,j,red)
                    if(i==0 or j==0 or i==7 or j==7):
                        if light_sensor.value == 1:
                            sense.set_pixel(i,j,white)
                        else:
                            sense.set_pixel(i,j,out)

        else:
            client.publish(mqtt_topic_led1,"true")
            client.publish(mqtt_topic_led2,"true")
            client.publish(mqtt_topic_led3,"true")

            for i in range(8):
                for j in range(8):
                    sense.set_pixel(i, j, green)
                    if(i==0 or j==0 or i==7 or j==7):
                        if light_sensor.value == 1:
                            sense.set_pixel(i,j,white)
                        else :
                            sense.set_pixel(i,j,out)

        print(round(heading,2))
        print(f"Direction: {direction}")
        print("Left: {:.2f} cm".format(distance1))
        print("Center: {:.2f} cm".format(distance2))
        print("Right: {:.2f} cm".format(round(distance3,2)))
        print("Humid(%) : ", round(humid,2))
        print("Temperature(oC) :", round((temp - 12),2))

        light = ""

        if light_sensor.value:
            print("Light : ON")
            light ="ON"
        else:
            print("Light : OFF")
            light = "OFF"

        # 센서 데이터 MQTT로 전송
        payload = {
            "Heading": round(heading,2),
            "Direction": direction,
            "Left":round(distance1,2),
            "Center":round(distance2,2),
            "Right": round(distance3,2),
            "Humid": round(humid,2),
            "Temp": round(temp - 12,2),  # 온도 보정
            "Light" : light,
            "Sig" : sig
        }

        client.publish(mqtt_topic1,str(payload))
        client.publish(mqtt_topic2,json.dumps(payload))

        client.publish(mqtt_topic_heading, round(heading,2))
        client.publish(mqtt_topic_direction, direction)
        client.publish(mqtt_topic_distance1, round(distance1,2))
        client.publish(mqtt_topic_distance2, round(distance2,2))
        client.publish(mqtt_topic_distance3, round(distance3,2))
        client.publish(mqtt_topic_humidity, round(humid,2))
        client.publish(mqtt_topic_temperature, round(temp-12,2))
        client.publish(mqtt_topic_light, light)

        current_time = datetime.now(korea_timezone)
        now_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        write_key = now_time

        result = firebase.put("/sensorData", write_key ,payload)
        print("Data successfully written. Key:", result)

        #time.sleep(0.5)

except KeyboardInterrupt:
    print("종료")

finally:
    for i in range(8):
        for j in range(8):
            sense.set_pixel(i, j, out)
    sensor1.close()
    sensor2.close()
    sensor3.close()
    client.disconnect()
