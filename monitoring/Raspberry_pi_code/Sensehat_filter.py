#방향을 구하기위해 칼만 필터를 구현했으나 오차가 커서 실패
#자이로와 가속도 센서로 상보필터를 구현했으나 오차가 커서 실패

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

        gyro = sense.get_gyroscope()
        gyro_x = gyro['pitch']
        gyro_y = gyro['roll']
        gyro_z = gyro['yaw']
        print(f"Gyro - X:{gyro_x}, Y:{gyro_y}, Z:{gyro_z}")

        accel = sense.get_accelerometer()
        accel_x = accel['pitch']
        accel_y = accel['roll']
        accel_z = accel['yaw']
        print(f"Accel - X:{accel_x}, Y:{accel_y}, Z:{accel_z}")

        ori = sense.get_orientation_degrees()
        x = ori['pitch']
        y = ori['roll']
        z = ori['yaw']
        print(f"Degree : X:{x}, Y:{y}, Z:{z} ")

        angleAcy = math.atan(-accel_x / math.sqrt((accel_y * accel_y) + accel_z * accel_z))
        angleAcy = angleAcy * 180 / 3.141592

        angleGyy = (gyro_y * 0.1 + angle)

        angle = 0.96 * angleGyy + 0.04 * angleAcy

        if distance2 <=5 :
            for i in range (8) :
                for j in range(8) :
                    sense.set_pixel(i,j,red)

        else :
            for i in range(8) :
                for j in range(8) :
                    sense.set_pixel(i,j,green)

        print(angle)
        print("Distance2: {:.2f} cm".format(distance2))
        print("Humidity(%) : ", humid)
        print("Temperature(oC) :", temp-8)

        time.sleep(1)

except KeyboardInterrupt:
    print("종료")

finally:
    for i in range(8) :
        for j in range(8) :
            sense.set_pixel(i,j,out)
