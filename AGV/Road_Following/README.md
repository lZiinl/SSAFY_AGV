# Road_Following

아래와 같은 Map이 주어지고 경로를 따라 이동하는 알고리즘, AI 학습 및 모델이 필요하다.

![331123035-925f2fac-51b1-43bb-a1dc-47c0ac428fdc](https://github.com/lZiinl/SSAFY_AGV/assets/149471946/74967f3b-9293-424c-802e-c056f5bedadb)


Road_Following을 구현하기 위해 AGV에 기본적으로 참조된 코드를 통해 데이터 수집, 학습, 확용을 해보았다.

# Data_Collection
 ## 1. 라이브러리 가져오기
 ```
    import traitlets
    import ipywidgets.widgets as widgets
    from IPython.display import display
    from jetbot import Robot, Camera, bgr8_to_jpeg

    from uuid import uuid1
    import os
    import json
    import glob
    import datetime
    import numpy as np
    import cv2
    import time
    from SCSCtrl import TTLServo

    # 카메라 세팅 및 로봇 객체 지정
    robot = Robot()
    TTLServo.servoAngleCtrl(5, 50, 1, 100)
 ```
 
 ## 2. AGV 제어를 위한 컨트롤러 추가하기
실제 AGV를 움직이면서 이미지를 캡쳐하기 위해 AGV를 움직이도록 하기 위한 버튼을 만들어 놓았다.
  ```
    # create buttons
    button_layout = widgets.Layout(width='100px', height='80px', align_self='center')
    stop_button = widgets.Button(description='stop', button_style='danger', layout=button_layout)
    forward_button = widgets.Button(description='forward', layout=button_layout)
    backward_button = widgets.Button(description='backward', layout=button_layout)
    left_button = widgets.Button(description='left', layout=button_layout)
    right_button = widgets.Button(description='right', layout=button_layout)
    # 레이아웃 생성 후,버튼 5개 생성

    # display buttons
    middle_box = widgets.HBox([left_button, stop_button, right_button], layout=widgets.Layout(align_self='center'))
    controls_box = widgets.VBox([forward_button, middle_box, backward_button])
    display(controls_box)
  ```
  ![image](https://github.com/homekepa/SSAFY_AGV/assets/91517560/91d3653d-db54-4386-81f4-81f5b9d99f21)


 ## 3. 컨트롤러에서 버튼 동작 시 호출하는 함수
 위에서 만들어진 버튼을 눌러서 AGV가 이동하는 지 확인을 한다.
```
    def stop(change): # 정지
    robot.stop()
    
def step_forward(change): # 전진
    robot.forward(0.4)

def step_backward(change): # 후진
    robot.backward(0.4)

def step_left(change): # 좌회전
    robot.left(0.3)
    time.sleep(0.5)
    robot.stop()

def step_right(change): # 우회전
    robot.right(0.3)
    time.sleep(0.5)
    robot.stop()

stop_button.on_click(stop)
forward_button.on_click(step_forward)
backward_button.on_click(step_backward)
left_button.on_click(step_left)
right_button.on_click(step_right)
```
## 4. 데이터 수집 경로 설정
 현제 디렉토리에 "dataset_xy_test"라는 폴더를 생성, 데이터를 수집하는 경로를 DATASET_DIR변수에 저장한다.
  ```
  DATASET_DIR = 'dataset_xy_test'
  
  try:
      os.makedirs(DATASET_DIR)
  except FileExistsError:
      print('Directories not created becasue they already exist')
  ```
## 5. 카메라 송출하기
 실제 카메라를 통해서 찍히는 영상을 송출해준다. 그리고 AGV 학습에 필요한 데이터 값 target(나아가야할 방향)위치를 표시하는 화면도 송출해준다. <br>
 두 개의 x축 실린더, y축 실린더를 이용해 target의 위치를 정할 수 있다.
 
 ```
 camera = Camera()
 
 image_widget = widgets.Image(format='jpeg', width=224, height=224)
 target_widget = widgets.Image(format='jpeg', width=224, height=224)
 
 x_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='x')
 y_slider = widgets.FloatSlider(min=-1.0, max=1.0, step=0.001, description='y')
 
 def display_xy(camera_image):
     image = np.copy(camera_image)
     x = x_slider.value
     y = y_slider.value
     x = int(x * 224 / 2 + 112)
     y = int(y * 224 / 2 + 112)
     image = cv2.circle(image, (x, y), 8, (0, 255, 0), 3)
     image = cv2.circle(image, (112, 224), 8, (0, 0,255), 3)
     image = cv2.line(image, (x,y), (112,224), (255,0,0), 3)
     jpeg_image = bgr8_to_jpeg(image)
     return jpeg_image
 
 time.sleep(1)
 traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
 traitlets.dlink((camera, 'value'), (target_widget, 'value'), transform=display_xy)
 display(widgets.HBox([image_widget, target_widget]))
 
 count_widget = widgets.IntText(description='count',
  value=len(glob.glob(os.path.join(DATASET_DIR, '*.jpg'))))
 save_button = widgets.Button(description='Save', button_style='success')
 
 display(widgets.VBox([x_slider, y_slider, count_widget, save_button]))
 ```
![image](https://github.com/homekepa/SSAFY_AGV/assets/91517560/e0c216f3-6add-4264-9277-3c74603a909c)
## 6. 데이터 수집하기
이제 위에 나온 이미지를 저장할 차례이다.
```
def xy_uuid(x, y):
    return 'xy_%03d_%03d_%s' % (x * 50 + 50, y * 50 + 50, uuid1())

def save_snapshot():
    uuid = xy_uuid(x_slider.value, y_slider.value)
    image_path = os.path.join(DATASET_DIR, uuid + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image_widget.value)
    count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))
    
save_button.on_click(lambda x: save_snapshot())
```
 수집하는 데이터는 다음과 같다.
 - Camera의 실시간 영상을 보고, AGV가 나아가야 하는 Target의 위치에 녹색 점을 둔다.
 - save버튼을 통해 save_snapshot()함수를 호출 데이터를 저장한다.
 - 수집한 데이터 파일은 dataset_xy_test 폴더에 저장되며, 파일의 이름은
   > \"xy_xValue_yValue_UUID.jpg\" 의 형식을 가진다. <br>
   >  여기서 xValue와 yValue는 '카메라 송출하기'의 실린더를 조정하여 내가 가고자하는 target(초록원)의 위치를 나타낸다.
 
 ![AGV_PJT](https://github.com/homekepa/SSAFY_AGV/assets/91517560/7121bb4e-b010-4948-a596-ca17a8d3c998)


---
약 3400장의 데이터 이미지를 수집했다.;;
![image](https://github.com/homekepa/SSAFY_AGV/assets/91517560/558bf705-c2f4-498d-a291-00d1aef370f8)

# Train Model
이제 위에서 수집한 데이터들을 학습시킬 차례이다. 
# Live Demo
