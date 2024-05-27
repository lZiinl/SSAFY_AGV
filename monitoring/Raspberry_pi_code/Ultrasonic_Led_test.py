from gpiozero import LED
import time
from gpiozero import DistanceSensor

# GPIO 핀 번호 설정
red_pin = 5  # LED1을 GPIO 18에 연결
green_pin = 6  # LED2를 GPIO 23에 연결
sensor1_trigger = 23
sensor1_echo = 24
sensor2_trigger = 15
sensor2_echo = 14
sensor3_trigger = 17
sensor3_echo = 27

# LED 객체 생성
red_led = LED(red_pin)
green_led = LED(green_pin)

# 초음파 센서 객체 생성
sensor1 = DistanceSensor(echo=sensor1_echo, trigger=sensor1_trigger)
sensor2 = DistanceSensor(echo=sensor2_echo, trigger=sensor2_trigger)
sensor3 = DistanceSensor(echo=sensor3_echo, trigger=sensor3_trigger)

try:
    while True:
        # 거리 측정
        distance1 = sensor1.distance * 100  # 거리를 센티미터로 변환
        distance2 = sensor2.distance * 100
        distance3 = sensor3.distance * 100

        print("Distance1: {:.2f} cm".format(distance1))
        print("Distance2: {:.2f} cm".format(distance2))
        print("Distance3: {:.2f} cm".format(distance3))

        # LED 제어
        if distance1 <= 5 or distance2 <= 5:
            red_led.on()
            green_led.off()
        else:
            red_led.off()
            green_led.on()

        time.sleep(1)

except KeyboardInterrupt:
    print("종료")

finally:
    red_led.off()
    green_led.off()
