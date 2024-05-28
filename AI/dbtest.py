import firebase_admin
from firebase_admin import credentials, db

# Firebase 서비스 계정 키 파일 경로
cred_path = './kfcproject-2fcc0-firebase-adminsdk-s7bwc-8baeeb05ff.json'

# Firebase 초기화
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://kfcproject-2fcc0-default-rtdb.firebaseio.com/'
})

# 최근 10개의 sensorData 읽기
ref = db.reference('sensorData')
data = ref.order_by_key().limit_to_last(10).get()

# 결과 출력
for key, value in data.items():
    print(f"{key}: {value}")

