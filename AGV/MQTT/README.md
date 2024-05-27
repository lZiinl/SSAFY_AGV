# MQTT
AGV와 다른 모듈, UI(Node-Red)와 통신을 위해 MQTT를 사용했다.

## Test코드
위의 코드를 통해 이미지를 송수신을 테스트할 수 있다.

MQTT라이브러리를 설치 참고 : https://www.youtube.com/watch?v=ZoCrEzADSS4

TOPIC : /agv0/image

<br>
송신 데이터는 카메라에서 촬영된 이미지를 .jpg로 변환하여 전송한다.
