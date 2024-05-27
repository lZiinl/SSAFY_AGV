#전체 코드 중 Firebase와 통신 부분만 추출
#전체 코드는 Monitoring의 raspberry pi 부분으로 이동

from firebase import firebase
from datetime import datetime
import pytz

# Firebase Realtime Database에 연결하는 Firebase 객체 생성
# project host url
firebase_url = "url"
firebase = firebase.FirebaseApplication(firebase_url, None)

current_time = datetime.now(korea_timezone)
now_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
write_key = now_time
result = firebase.put("/sensorData", write_key ,payload)
print("Data successfully written. Key:", result)
