from sense_hat import SenseHat
import time
from gpiozero import DistanceSensor
import math

# GPIO 핀 번호 설정

#sensor1_trigger = 17
#sensor1_echo = 27
sensor2_trigger = 15
sensor2_echo = 14

sense = SenseHat()

# 초음파 센서 객체 생성
#sensor1 = DistanceSensor(echo=sensor1_echo, trigger=sensor1_trigger)
sensor2 = DistanceSensor(echo=sensor2_echo, trigger=sensor2_trigger)

#led brightness
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
white = (255,255,255)
out = (0,0,0)
angle = 0


try:
    while True:
        # 거리 측정
        #distance1 = sensor1.distance * 100  # 거리를 센티미터로 변환
        distance2 = sensor2.distance * 100

        #print("Distance1: {:.2f} cm".format(distance1))
        #print("Distance2: {:.2f} cm".format(distance2))

        humid = sense.get_humidity()
        temp = sense.get_temperature()



        if distance2 <=5 :
            for i in range (8) :
                for j in range(8) :
                    sense.set_pixel(i,j,red)

        else :
            for i in range(8) :
                for j in range(8) :
                    sense.set_pixel(i,j,green)


        print("Distance2: {:.2f} cm".format(distance2))
        print("Humidity(%) : ", humid)
        print("Temperature(oC) :", temp-8)
        
        com = sense.get_compass_raw()
        print(f'[{com["x"]:5.1f}] - ', end = '')
        print(f'[{com["y"]:5.1f}] - ', end = '')
        print(f'[{com["z"]:5.1f}]', end = '')
        print()

        time.sleep(1)

except KeyboardInterrupt:
    print("종료")

finally:
    for i in range(8) :
        for j in range(8) :
            sense.set_pixel(i,j,out)
