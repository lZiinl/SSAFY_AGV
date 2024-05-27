# AGV_Project

앞서 작업한 모델과, HSV값, MQTT를 이용해서 합쳐서 동작을 구성했다.


## 라이브러리 불러오기
```
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

from IPython.display import display
import ipywidgets
import ipywidgets.widgets as widgets
import traitlets
from jetbot import Robot, Camera, bgr8_to_jpeg

import paho.mqtt.client as mqtt
import threading
import time
from SCSCtrl import TTLServo
```

## 모델 불러오기 & 카메라, 로봇 객체 초기화
```
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('best_steering_model_xy_test.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
print('model load success')
```

MQTT도 포함하고 있다. <br>
execute()는 이미지를 전송하는 부분을 담당하고 agv0/image 토픽에 송신 한다.

on_message()는 UI에서 송신하는 값을 수신하는 부분이다.

Object Detection값은 RPI에서 파악한 전방 뮬체가 감지 되었을 시 수신되는 키워드로 자동상황일 때 수동으로 전환이 된다.(수동 상태일 때는 전환되지 않는다.)<br>

mode_change값은 수동에서 자동 혹은 자동에서 수동으로 전환하는 키워드이다. <br>

_Color_(red, blue, orange, green, purple, yellow)값이 들어 왔을 떄 목적지를 전환하고 자동으로 전환한다. (자동상태에서 전환되지 않는다.) <br>


```
#로봇 초기객체 초기화
robot = Robot()
#모터 초기화(Arm모터)
TTLServo.xyInputSmooth(100, 0, 1)
TTLServo.servoAngleCtrl(5, 50, 1, 100)
TTLServo.servoAngleCtrl(4, 10, 1, 100)
print('robot arm Ready!')

#카메라 객체 초기화
# MQTT 클라이언트 설정
broker_address = "127.0.0.1"
topic = "agv0/image"
subscribe_topic = "agv0/message"

client = mqtt.Client()
client.connect(broker_address)

# JetBot 카메라 설정
#camera = Camera.instance(width=300, height=300)
camera = Camera()
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

# MQTT 수신 메시지 처리 함수
def on_message(client, userdata, message):
    msg_payload = message.payload.decode()
    # 전방 물체 감지 확인후 수동 전환
    if msg_payload == "Object Detection" and startBtn.description == "Stop":
        print(msg_payload)
        start(True)
        
    # 모드 전환
    elif msg_payload == "mode_change":
        print(msg_payload)
        start(True)
        
    elif msg_payload == "red" and startBtn.description == "Start":
        print(msg_payload)
        change_color("red")
        start(True)

    elif msg_payload == "blue" and startBtn.description == "Start":
        print(msg_payload)
        change_color("blue")
        start(True)
        
    elif msg_payload == "orange" and startBtn.description == "Start":
        print(msg_payload)
        change_color("orange")
        start(True)
        
    elif msg_payload == "green" and startBtn.description == "Start":
        print(msg_payload)
        change_color("green")
        start(True)
        
    elif msg_payload == "purple" and startBtn.description == "Start":
        print(msg_payload)
        change_color("purple")
        start(True)
        
    elif msg_payload == "yellow" and startBtn.description == "Start":
        print(msg_payload)
        change_color("yellow")
        start(True)
    
client.on_message = on_message
client.subscribe(subscribe_topic)
client.loop_start()
traitlets.dlink((camera, 'value'), (image, 'value'), transform=execute)
```


## 색인지 범위 지정
Color_Recognition에서 측정한 값을 삽입한다. (특정위치에서 측정된 값입니다)

change_color()함수를 통해 목적지의 값을 바꾸준다. (default red)
```
#set areaA
areaA = 'red'
colors = [
        {'name': 'red', 'lower': np.array([0, 80, 140]), 'upper': np.array([10, 180, 200])},
        {'name': 'green', 'lower': np.array([50, 70, 120]), 'upper': np.array([80,180, 200])},
        {'name': 'blue', 'lower': np.array([80, 50, 50]), 'upper': np.array([110, 250, 200])},
        {'name': 'purple', 'lower': np.array([105, 60, 110]), 'upper': np.array([130, 165, 154])},
        {'name': 'yellow', 'lower': np.array([25, 90, 160]), 'upper': np.array([30, 200, 210])},
        {'name': 'orange', 'lower': np.array([10, 100, 140]), 'upper': np.array([20, 180, 200])},
]

areaA_color = next((color for color in colors if color['name'] == areaA), None)
areaAlbl.value = areaA_color['name']

findArea = areaA
goallbl.value = findArea

#frame 크기와 카메라 중심점 좌표 설정
frame_width = 120
frame_height = 320
camera_center_X = int(frame_width/2)
camera_center_Y = int(frame_height/2)

#AreaFind() 용 리스트와 변수
colorHSVvalueList = []
max_len = 20

def change_color(color):
    global areaA, areaA_color
    areaA = color

    areaA_color = next((color for color in colors if color['name'] == areaA), None)
    areaAlbl.value = areaA_color['name']

    findArea = areaA
    goallbl.value = findArea


#2개 thread 용 객체 변수 생성
roadFinding = None
goalFinding = None
```

--생략--

## Joystick용 스레드 생성
바퀴가 움직이면서 팔을 움직일 수 있어야 한다. <br>

```
class ArmCtrlThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(ArmCtrlThread, self).__init__(*args, **kwargs)
        self.moveXin = 0
        self.moveXde = 0
        self.moveYin = 0
        self.moveYde = 0
        self.th_flag = True
        print("수동 암")
        
    def st(self):
        self.th_flag = True

    def stop(self):
        self.th_flag = False

    def run(self):
        while self.th_flag:
            
            armCtrl(self.moveXin, self.moveXde, self.moveYin, self.moveYde)
            time.sleep(movingTime)
        robot.stop()
        print("ArmCtrlThread End")


class inputThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(inputThread, self).__init__(*args, **kwargs)
        self.th_flag = True
        print("수동 이동")
    
    def st(self):
        self.th_flag = True

    def stop(self):
        self.th_flag = False

    def run(self):
        
        while self.th_flag:
            evbuf = jsdev.read(8)
            if evbuf:
                time, value, type, number = struct.unpack('IhBB', evbuf)
                if type & 0x80:
                    print("(initial)", end="")
                if type & 0x01:
                    button = button_map[number]
                    if button:
                        grabCtrlCommand(button, value)
                        armCtrlCommand(button, value)
                if type & 0x02:
                    axis = axis_map[number]
                    if axis:
                        moveSmoothCtrl(axis, value / 32767.0)
        robot.stop()
        print("inputThread End")
```

## WorkingAreaFind() 클래스 생성하기 / 색깔 인식
색 인식과 Road detection은 동시에 동작한다. <br>
스레드로 생성하여 동작한다.
```
class WorkingAreaFind(threading.Thread):
    flag = 1
    def __init__(self):
        super().__init__()
        self.th_flag=True
        self.imageInput = 0
        flaglbl.value = str(WorkingAreaFind.flag)
        self._stop_event = threading.Event()
        
    def st(self):
        self.th_flag = True

    def run(self):
        while self.th_flag:
            self.imageInput = camera.value
            #BGR to HSV
            hsv = cv2.cvtColor(self.imageInput, cv2.COLOR_BGR2HSV)
            #blur
            hsv = cv2.blur(hsv, (15, 15))
            self.colorRecog(hsv)

            #areaA, areaB Color searching
            areaA_mask = cv2.inRange(hsv, areaA_color['lower'], areaA_color['upper'])
            areaA_mask = cv2.erode(areaA_mask, None, iterations=2)
            areaA_mask = cv2.dilate(areaA_mask, None, iterations=2)


            # 해당 영역에 대한 윤곽선 따기
            AContours, _ = cv2.findContours(areaA_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #A영역
            if AContours and WorkingAreaFind.flag == 1:
                self.findCenter(areaA, AContours)

            #두 영역 모두 못찾았다면, 찾아가는 중이다.
            else:
                cv2.putText(self.imageInput, "Finding...", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
                image_widget.value = bgr8_to_jpeg(self.imageInput)
            time.sleep(0.1)
            

    #name : A, B 구분용도, Contours 각 영역의 윤곽선 값
    def findCenter(self, name, Contours):
        c = max(Contours, key=cv2.contourArea)
        ((box_x, box_y), radius) = cv2.minEnclosingCircle(c)

        X = int(box_x)
        Y = int(box_y)

        error_Y = abs(camera_center_Y - Y)
        error_X = abs(camera_center_X - X)

        if error_Y < 15 and error_X < 15:
            #A영역이 가까이 오게됨
            if name == areaA and self.flag == 1:
                robot.stop()
                #test 코드
                areaAlbl.value = areaA + " Goal!"
            

                ###########################
                print("start_go")
                start(True)
               
                ###########################
                print("Road Following End")

                
        image_widget.value = bgr8_to_jpeg(self.imageInput)

    def colorRecog(self, hsv):
        # Center Pixel 의 hsv 값 읽어오기
        hsvValue = hsv[int(frame_height / 2), int(frame_width / 2)]
            
        # data 20개 모아서, 최대, 최소 값 구하기
        colorHSVvalueList.append(hsvValue)
        if len(colorHSVvalueList) > max_len:
            del colorHSVvalueList[0]
                
        max_h, max_s, max_v = np.maximum.reduce(colorHSVvalueList)
        min_h, min_s, min_v = np.minimum.reduce(colorHSVvalueList)
        # Center Pixel 주위에 20x20 크기의 사각형 그리기
        rect_s = 20
        cv2.rectangle(self.imageInput,
                    (int(frame_width / 2) - int(rect_s / 2), int(frame_height / 2) - int(rect_s / 2)),
                    (int(frame_width / 2) + int(rect_s / 2), int(frame_height / 2) + int(rect_s / 2)),
                    (0, 0, 255), 1)
            
        # max, min value 표시
        cv2.putText(self.imageInput, f'max_HSV:{ max_h, max_s, max_v }', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(self.imageInput, f'min_HSV:{ min_h, min_s, min_v }', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        colorlblMax.value = str(max_h) + "," + str(max_s) + "," + str(max_v)
        colorlblMin.value = str(min_h) + "," + str(min_s) + "," + str(min_v) 
              

    def stop(self):
        self.th_flag = False
        self._stop_event.set()
        robot.stop()
```

## RobotMoving() 클래스 생성하기 / road detection
WorkingAreaFind()와 같이 동작해야하기에 스레드 생성을 한다.
```
class RobotMoving(threading.Thread):
    print('robot moving!')
    def __init__(self):
        super().__init__()
        self.th_flag = True
        self.angle = 0.0
        self.angle_last = 0.0
        self._stop_event = threading.Event()
        
    def st(self):
        self.th_flag = True

    def run(self):
        while self.th_flag:
            image = camera.value
            xy = model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
            x = xy[0]
            y = (0.5 - xy[1]) / 2.0

            x_slider.value = x
            y_slider.value = y

            speed_slider.value = speed_gain_slider.value
            image_widget.value = bgr8_to_jpeg(image)

            self.angle = np.arctan2(x, y)

            if not self.th_flag:
                break

            pid = self.angle * steering_gain_slider.value + (self.angle - self.angle_last) * steering_dgain_slider.value
            self.angle_last = self.angle

            steering_slider.value = pid + steering_bias_slider.value

            robot.left_motor.value = max(min(speed_slider.value + steering_slider.value, 1.0), 0.0)
            robot.right_motor.value = max(min(speed_slider.value - steering_slider.value, 1.0), 0.0)
            time.sleep(0.1)
        robot.stop()
        print("Area Finding End")

    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device).half()
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def stop(self):
        self.th_flag = False
        self._stop_event.set()
        robot.stop()
```

## Start 버튼 함수 정의하고 바인딩하기
프로그램 첫 동작시 수동상태로 시작한다.

start()는 startBtn버튼에 바인딩되어 동작하며, 버튼의 값에 따라 동작을 다르게 한다. 

"Start" : 자동상태로 전환하는 버튼이다.
자동 상태에서는 RobotMoving(), WorkingAreaFind() 스레드가 동작하며 <br>
ArmCtrlThread(), inputThread()의 스레드는 종료된다. <br>


"Stop" : 수동상태로 전환하는 버튼이다.
수동 상태에서는 ArmCtrlThread(), inputThread()의 스레드는 동작하며 <br>
RobotMoving(), WorkingAreaFind() 스레드는 종료된다.

```
# 시작하자마자 기본 수동상태 전환
ctrlArm = ArmCtrlThread()
ctrlArm.start()
inputThreading = inputThread()
inputThreading.start()
#roadFinding = RobotMoving()
#goalFinding = WorkingAreaFind()


def start(change):
    global modeBtn, camera_link, goalFinding, roadFinding, areaA, findArea, goallbl, ctrlArm, inputThreading
    #Start -> Auto
    if startBtn.description == "Start":
        modeBtn.value = "Auto"
        modeBtn.disabled = True

        for i in manual_btnlst:
            i.disabled = True

        goalFinding = WorkingAreaFind()
        goalFinding.start()
        roadFinding = RobotMoving()
        roadFinding.start()

        startBtn.button_style = "warning"
        startBtn.description = "Stop"
        
        # 기존 스레드 종료
        ctrlArm.stop()
        #ctrlArm.join()
        inputThreading.stop()
        #inputThreadinga.join()
        # 카메라 링크 해제
        camera_link.unlink()

    elif startBtn.description == "Stop":
        roadFinding.stop()
        goalFinding.stop()
        print("Warning!!")

        ctrlArm = ArmCtrlThread()
        ctrlArm.start()
        inputThreading = inputThread()
        inputThreading.start()

        startBtn.button_style = "info"
        startBtn.description = "Start"
        for i in manual_btnlst:
            i.disabled = False
        modeBtn.value = "Auto"
        modeBtn.disabled = False
        camera_link = traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
        
startBtn.on_click(start)
```

WorkingAreaFind()에서 목적지에 도작하면 start()함수를 호출하며 수동으로 전환한다. <br>
버튼의 상태는 Start와 stop인 두 상태만 존재하며, 토글로 작동한다. <br>

