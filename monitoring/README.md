# 라즈베리파이 및 Node-red 기반의 모니터링 시스템<br>
![image](https://github.com/lZiinl/SSAFY_AGV/assets/149471946/c04bddd3-7288-400d-8e2c-057d42c55f01)
<br>

## 시스템 개요<br>

### 라즈베리 파이 기반 센서 데이터 측정 및 led 제어
1. 라즈베리파이5를 사용하여 센스헷, 조도센서, 초음파 센서 3개로 작업 환경 측정 및 장애물 인식<br>
2. 10cm 이내의 장애물을 인식한 경우 센스헷의 LED를 빨간색으로 표시, 없는 경우 초록색으로 표시<br>
3. 장애물 존재 시 AGV를 수동 모드로 전환<br>
4. 조도에 따른 조도에 따른 Light on/off<br>

### Node-red 기반의 모니터링
1. 측정된 센서 데이터를 MQTT를 통해 브로커 서버인 PC로 보내고, PC에서 Node-red로 구성된 UI에 표현<br>
2. 잿슨 나노에 연결된 카메라 정보를 MQTT를 통해 브로커 서버인 PC로 보내고, PC에서 Node-red로 구성된 UI에 표현<br>




## 시스템 Block Diagram<br>
![그림4](https://github.com/lZiinl/SSAFY_AGV/assets/149471946/44434e31-8cba-4582-9bea-c74398fed620)

## 시스템 개요<br>
