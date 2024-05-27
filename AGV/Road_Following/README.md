# Road_Following

아래와 같은 Map이 주어지고 경로를 따라 이동하는 알고리즘, AI 학습 및 모델이 필요하다.

![331123035-925f2fac-51b1-43bb-a1dc-47c0ac428fdc](https://github.com/lZiinl/SSAFY_AGV/assets/149471946/74967f3b-9293-424c-802e-c056f5bedadb)


Road_Following을 구현하기 위해 AGV에 기본적으로 참조된 코드를 통해 데이터 수집, 학습, 동작을 했다.

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
실제 AGV를 움직이면서 이미지를 캡쳐하기 위해 AGV를 움직이기 위한 버튼을 만들어 놓았다. (직접 옮겨서 움직이는건 힘드니깐)
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
위에서 만들어진 버튼을 눌러서 AGV가 이동하는지 확인을 한다. 

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

   > ### + 추가
   > 데이터를 모아본 결과 직선으로 이동하는 데이터와 이탈했을시 복귀하는 데이터(AGV의 중앙과 떨어지는 데이터) <br>
   > 이 데이터의 비율은 7(직선) : 3(복귀) 가 적합해 보인다.
   > 직접 데이터를 삽입하고 확인하는 과정에서 Z자로 이동하는 경우가 많았고 코너에서 잘 찾아가지만 로 이탈이 잦았다.
   > 직선데이터를 더 추가하고 학습하면서 해소되기도 했다. 
   
 ![AGV_PJT](https://github.com/homekepa/SSAFY_AGV/assets/91517560/7121bb4e-b010-4948-a596-ca17a8d3c998)


---
약 3400장의 데이터 이미지를 수집했다.;;
![image](https://github.com/homekepa/SSAFY_AGV/assets/91517560/558bf705-c2f4-498d-a291-00d1aef370f8)


# Train Model
이제 위에서 수집한 데이터들을 학습시킬 차례이다.
- Colab에서 데이터 학습을 진행했습니다. 
## 1. 라이브러리 가져오기
```
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
```
## 2. DataSet 인스턴스 생성하기
여기에서는 torch.utils.data.Dataset 클래스를 구현하는 사용자 정의 클래스를 생성한다.  
이 클래스는 len 및 getitem 함수를 구현한다. 이 클래스는 이미지를 로드하고 이미지 파일 이름에서 x, y 값을 파싱하는 역할을 한다.   
torch.utils.data.Dataset 클래스를 구현했으므로 torch 데이터 유틸리티를 모두 사용할 수 있다.  

```
DATASET_DIR = 'dataset_xy_test'

#image 이름으로 저장된 x 값을 읽어 오는 함수
def get_x(path):
    """Gets the x value from the image filename"""
    return (float(int(path[3:6])) - 50.0) / 50.0

#image 이름으로 저장된 y 값을 읽어 오는 함수
def get_y(path):
    """Gets the y value from the image filename"""
    return (float(int(path[7:10])) - 50.0) / 50.0


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))
        
        if float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x
        
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor([x, y]).float()
    
dataset = XYDataset(DATASET_DIR, random_hflips=False)
```
## 3. DataSet 분할하기
데이터셋을 읽은 후에는 데이터셋을 훈련 세트와 테스트 세트로 분할한다.  
이 코드에서는 훈련 세트와 테스트 세트를 90%-10%로 분할한다.   
테스트 세트는 훈련한 모델의 정확도를 검증하는 데 사용된다.
```
test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
```
## 4. DataLoader 생성하기
데이터를 일괄 처리로 로드하기 위해 DataLoader 클래스를 사용하여 데이터 로더를 생성한다.  
이를 통해 데이터를 일괄 처리로 로드하고 데이터를 섞고, 여러 개의 서브프로세스를 사용할 수 있다.  
이 예에서는 배치 크기를 64로 사용한다.  
배치 크기는 GPU의 사용 가능한 메모리에 따라 결정되며 모델의 정확도에 영향을 줄 수 있다.
```
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```
## 5. model 정의하기
PyTorch TorchVision에서 제공하는 ResNet-18 모델을 사용한다.
<br>
전이 학습(transfer learning)이라는 프로세스에서, 수백만 장의 이미지로 훈련된 사전 훈련된 모델을 다시 사용하여 가능한 매우 적은 데이터로 이루어진 새로운 작업에 활용할 수 있다.  
- ResNet-18 상세설명 : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py 
- 전이학습에 대한 유투브 설명: https://www.youtube.com/watch?v=yofjFQddwHE 

기본적으로, ResNet 모델은 fully connected (fc) 최종 레이어를 가지고 있으며, 입력 특성 수로 512를, 회귀를 위해 출력 특성 수로 1을 사용할 것이다.  
하지만, 우리는 x,y 두 개의 값을 도출해야 하기 때문에, 마지막 은닉층에 레이어를 하나 추가해서, 2개의 output 데이터가 나오도록 할 예정이다.  

마지막으로, 모델을 GPU에서 실행할 수 있도록 전송한다.
```
model = models.resnet18(pretrained=True)

model.fc = torch.nn.Linear(512, 2)
device = torch.device('cuda')
model = model.to(device)
```

## 6. model 훈련하기
손실이 감소되면 최상의 모델을 저장하기 위해 50 에포크 동안 훈련한다.  
훈련을 모두 마치면, "Success" 가 출력됩니다.
```
NUM_EPOCHS = 50
BEST_MODEL_PATH = 'best_steering_model_xy_test.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += float(loss)
    test_loss /= len(test_loader)
    
    print('%f, %f' % (train_loss, test_loss))
    if test_loss < best_loss:
        #colab에서 model을 학습할 경우 아래 옵션을 추가한 코드를 실행해야 한다.
        #torch.save(model.state_dict(), BEST_MODEL_PATH,_use_new_zipfile_serialization=False)
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss
print('success')
```
모델이 훈련되면 best_steering_model_xy.pth 파일이 생성된다.
colab에서 약 3400장 학습하는 하는데 걸린 시간 약 30분 걸렸다.

# Live Demo
