from gpiozero import DigitalInputDevice
import time

# GPIO 핀 번호 설정
LIGHT_SENSOR_PIN = 16

# 조도 센서 초기화
light_sensor = DigitalInputDevice(LIGHT_SENSOR_PIN)

try:
    while True:
        # 센서 상태 읽기
        print(light_sensor.value)
        if light_sensor.value:
            print("빛이 없습니다")
        else:
            print("빛이 있습니다")
        
        # 1초 대기
        time.sleep(1)

except KeyboardInterrupt:
    print("프로그램 종료")

