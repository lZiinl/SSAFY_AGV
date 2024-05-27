# Road_Following

아래와 같은 Map이 주어지고 경로를 따라 이동하는 알고리즘, AI이 필요하다.

![331123035-925f2fac-51b1-43bb-a1dc-47c0ac428fdc](https://github.com/lZiinl/SSAFY_AGV/assets/149471946/74967f3b-9293-424c-802e-c056f5bedadb)


Road_Following을 구현하기 위해 AGV에 기본적으로 참조된 코드를 통해 데이터 수집, 학습, 확용을 해보았다.

# Data_Collection
 1. 라이브러리 가져오기
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
 2. AGV 제어를 위한 컨트롤러 추가하기
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
  

 3. 컨트롤러에서 버튼 동작 시 호출하는 함수
    ```
    DATASET_DIR = 'dataset_xy_test'

    try:
        os.makedirs(DATASET_DIR)
    except FileExistsError:
        print('Directories not created becasue they already exist')
    ```
 4. 카메라 송출하기



# Train Model

# Live Demo