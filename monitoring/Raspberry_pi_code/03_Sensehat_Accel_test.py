from sense_hat import SenseHat
from time import sleep

sense = SenseHat()

while True:
    humid = sense.get_humidity()
    temp = sense.get_temperature()
    
    accel = sense.get_accelerometer()
    x = accel['pitch']
    y = accel['roll']
    z = accel['yaw']
    
    print("Humidity(%) : ", humid)
    print("Temperature(oC) :", temp-8)
    print(f"Accel - X:{x}, Y:{y}, Z:{z}")

    sleep(1)
