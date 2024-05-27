from sense_hat import SenseHat
import time
from gpiozero import DistanceSensor
import math

# GPIO 핀 번호 설정

#sensor1_trigger = 27
#sensor1_echo = 17
#sensor2_trigger = 15
#sensor2_echo = 14
sensor3_trigger = 21
sensor3_echo = 20

sense = SenseHat()

# 초음파 센서 객체 생성
#sensor1 = DistanceSensor(echo=sensor1_echo, trigger=sensor1_trigger)
#sensor2 = DistanceSensor(echo=sensor2_echo, trigger=sensor2_trigger)
sensor3 = DistanceSensor(echo=sensor3_echo, trigger=sensor3_trigger)

#led brightness
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
white = (255,255,255)
out = (0,0,0)
angle = 0

def get_heading():
    # Get the orientation data
    orientation = sense.get_orientation_degrees()
    pitch = orientation['pitch']
    roll = orientation['roll']
    yaw = orientation['yaw']

    # Print the yaw angle
    print(f"Pitch: {pitch:.2f} Roll: {roll:.2f} Yaw: {yaw:.2f}")
    return yaw

try:
    while True:
        # 거리 측정
        #distance1 = sensor1.distance * 100  # 거리를 센티미터로 변환
        #distance2 = sensor2.distance * 100
        distance3 = sensor3.distance * 100

        humid = sense.get_humidity()
        temp = sense.get_temperature()

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

        if distance3 <=5 :
            for i in range (8) :
                for j in range(8) :
                    sense.set_pixel(i,j,red)

        else :
            for i in range(8) :
                for j in range(8) :
                    sense.set_pixel(i,j,green)

        print(heading)
        print(f"Direction: {direction}")
        #print("Distance1: {:.2f} cm".format(distance1))
        #print("Distance2: {:.2f} cm".format(distance2))
        print("Distance3: {:.2f} cm".format(distance3))
        print("Humidity(%) : ", humid)
        print("Temperature(oC) :", temp-8)

        time.sleep(1)

except KeyboardInterrupt:
    print("종료")

finally:
    for i in range(8) :
        for j in range(8) :
            sense.set_pixel(i,j,out)

    #sensor1.close()
    #sensor2.close()
    sensor3.close()

