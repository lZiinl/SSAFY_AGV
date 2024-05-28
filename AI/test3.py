import firebase_admin
from firebase_admin import credentials, db
import openai
import json

# Firebase 서비스 계정 키 파일 경로
cred_path = ''

# Firebase 초기화
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://kfcproject-2fcc0-default-rtdb.firebaseio.com/'
})

# 최근 10개의 sensorData 읽기
ref = db.reference('sensorData')
data = ref.order_by_key().limit_to_last(10).get()

# 데이터 출력
print("Data from Firebase:")
print(json.dumps(data, indent=2))

# 데이터를 JSON 문자열로 변환
data_json = json.dumps(data)

# OpenAI API 키 설정
openai.api_key = ""

# OpenAI API를 통해 평균 계산 요청
prompt = (
    f"Calculate the average temperature and humidity from the following data:\n"
    f"{data_json}\n"
)

response = openai.chat.completions.create(
    model="gpt-4-1106-preview",
                messages=[
                    {
                        "role" : "user",
                        "content" : prompt,
                        }
                    ],
                # prompt=prompt,
                max_tokens=2000 
                )

# OpenAI API 응답 출력
print("\nOpenAI API Response:")
print(response)

