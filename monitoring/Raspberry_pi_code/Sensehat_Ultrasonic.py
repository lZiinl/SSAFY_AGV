#센스헷의 온습도, 각도, 초음파 센서 융합 1


import time
from gpiozero import DistanceSensor
from sense_hat import SenseHat

# GPIO 핀 번호 설정

#sensor1_trigger = 17
#sensor1_echo = 27
sensor2_trigger = 15
sensor2_echo = 14

sense = SenseHat()

# 초음파 센서 객체 생성
#sensor1 = DistanceSensor(echo=sensor1_echo, trigger=sensor1_trigger)
sensor2 = DistanceSensor(echo=sensor2_echo, trigger=sensor2_trigger)

try:
    while True:
        # 거리 측정
        #distance1 = sensor1.distance * 100  # 거리를 센티미터로 변환
        distance2 = sensor2.distance * 100

        #print("Distance1: {:.2f} cm".format(distance1))
        #print("Distance2: {:.2f} cm".format(distance2))
        
        humid = sense.get_humidity()
        temp = sense.get_temperature()
    
        ori = sense.get_orientation_degrees()
        x = ori['pitch']
        y = ori['roll']
        z = ori['yaw']
        
        print("Distance2: {:.2f} cm".format(distance2))
        print("Humidity(%) : ", humid)
        print("Temperature(oC) :", temp-8)
        print(f"Degree : X:{x}, Y:{y}, Z:{z} ")

        time.sleep(1)

except KeyboardInterrupt:
    print("종료")
