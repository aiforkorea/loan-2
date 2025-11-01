# apps/dashboard/gen_data.py
import os
import csv
import sys 
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import random
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
from apps.dbmodels import User, UserType
from apps.iris.dbmodels import IrisResult, IrisClassType

# 현재 파일의 디렉토리
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV 파일 경로
PREDICTION_CSV_PATH = os.path.join(CURRENT_DIR, 'prediction_results.txt')
IRIS_CSV_PATH = os.path.join(CURRENT_DIR, 'iris_results.txt')

# 설정 상수
NUM_DAYS = 60 # 1
ITEMS_PER_CLASS_PER_DAY = 10 # 2
SERVICE_ID = 1
API_KEY_ID = 1
MODEL_VERSION = '1.0'
CONFIRM_RATE = 0.85 # 85% 확률로 predicted_class와 동일하게 confirmed_class 설정

# 일반 사용자 ID 목록
USER_IDS = [3, 4]
# IRIS 클래스 목록
IRIS_CLASSES = [e.value for e in IrisClassType]
# 클래스별 평균 및 표준편차 (가상의 IRIS 데이터)
IRIS_MEANS_STDEVS = {
    'setosa': ([5.01, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.10]),
    'versicolor': ([5.94, 2.77, 4.26, 1.33], [0.52, 0.31, 0.47, 0.20]),
    'virginica': ([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
}

def generate_iris_data(predicted_class):
    """지정된 클래스를 기반으로 가상의 IRIS 측정값 생성"""
    means, stdevs = IRIS_MEANS_STDEVS.get(predicted_class, IRIS_MEANS_STDEVS['setosa'])
    return [
        round(random.gauss(means[i], stdevs[i]), 2) for i in range(4)
    ]

def create_users_data():
    """DB에 추가할 사용자 데이터를 생성합니다."""
    # 암호는 '1'을 사용하여 해시 생성
    password_input = '1'
    hashed_password = generate_password_hash(password_input)

    users_data = [
        # 전문가 (expert)
        {
            'id': 2,
            'username': 'kdhong2',
            'email': 'kdhong2@example.com',
            'password_hash': hashed_password,
            'user_type': UserType.EXPERT,
            'is_active': True,
            'is_deleted': False,
            'match_status': 'unassigned',
        },
        # 일반 사용자 1 (user)
        {
            'id': 3,
            'username': 'kdhong3',
            'email': 'kdhong3@example.com',
            'password_hash': hashed_password,
            'user_type': UserType.USER,
            'is_active': True,
            'is_deleted': False,
            'match_status': 'unassigned',
        },
        # 일반 사용자 2 (user)
        {
            'id': 4,
            'username': 'kdhong4',
            'email': 'kdhong4@example.com',
            'password_hash': hashed_password,
            'user_type': UserType.USER,
            'is_active': True,
            'is_deleted': False,
            'match_status': 'unassigned',
        },
    ]
    
    # DB에 사용자 객체를 생성합니다. (실제 DB에 추가하는 로직은 DB 연결 설정 후 별도 실행)
    user_objects = []
    for data in users_data:
        # User 모델의 `id` 필드를 직접 할당하고, `created_at` 등 타임스탬프는 현재 시각을 사용하도록 설정
        user = User(
            id=data['id'],
            username=data['username'],
            email=data['email'],
            password_hash=data['password_hash'],
            user_type=data['user_type'],
            is_active=data['is_active'],
            is_deleted=data['is_deleted'],
            match_status=data['match_status'],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        user_objects.append(user)
    
    # 생성된 사용자 정보 출력 (DB 입력 확인용)
    print("--- 생성된 사용자 정보 (DB 입력 대기 중) ---")
    for user in user_objects:
        print(f"ID: {user.id}, Username: {user.username}, Type: {user.user_type.value}, Email: {user.email}")
    print("---------------------------------------")

    return user_objects

def generate_csv_data():
    """IRIS 예측 결과 및 상세 데이터를 CSV 파일 형태로 생성합니다."""
    # 기존 파일 삭제 로직 (생략)
    if os.path.exists(PREDICTION_CSV_PATH):
        os.remove(PREDICTION_CSV_PATH)
        print(f"기존 파일 삭제: {PREDICTION_CSV_PATH}")
    if os.path.exists(IRIS_CSV_PATH):
        os.remove(IRIS_CSV_PATH)
        print(f"기존 파일 삭제: {IRIS_CSV_PATH}")

    # CSV 파일 헤더 정의 (생략)
    prediction_header = [
        "id", "user_id", "service_id", "api_key_id", "predicted_class", "is_deleted",
        "model_version", "confirmed_class", "confirm", "created_at", "confirmed_at", "type"
    ]
    iris_header = [
        "id", "sepal_length", "sepal_width", "petal_length", "petal_width", "redundancy"
    ]

    prediction_results = []
    iris_results = []
    current_id = 1
    start_date = datetime.now() - timedelta(days=NUM_DAYS)

    # 항목 수의 최소/최대 범위 계산
    base_items = ITEMS_PER_CLASS_PER_DAY  # 10
    # 수정된 부분: 최소 80%, 최대 120% 계산
    min_items = max(1, int(base_items * 0.80)) # 최소 1개 보장
    max_items = int(base_items * 1.20)
    # 현재 설정(ITEMS_PER_CLASS_PER_DAY=10)의 경우, min_items = 9, max_items = 11

    print(f"--- {NUM_DAYS}일간의 가상 데이터 생성 시작 (일일 항목 수: {min_items} ~ {max_items} 범위) ---")

    for day in range(NUM_DAYS):
        current_date = start_date + timedelta(days=day)
        for user_id in USER_IDS:
            for iris_class in IRIS_CLASSES:
                # 최소 80%에서 최대 120% 사이에서 랜덤하게 항목 수를 결정
                items_to_generate = random.randint(min_items, max_items)
                for item in range(items_to_generate):
                    # 1. PredictionResult 데이터 생성
                    created_at = current_date + timedelta(hours=random.randint(9, 17), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
                    # confirmed_class 및 confirm 설정 (생략)
                    confirmed_class = None
                    confirm = 0
                    if random.random() < CONFIRM_RATE:
                        confirmed_class = iris_class
                        confirm = 1
                        confirmed_at = created_at + timedelta(seconds=random.randint(10, 3600))
                    else:
                        other_classes = [c for c in IRIS_CLASSES if c != iris_class]
                        confirmed_class = random.choice(other_classes)
                        confirm = 1
                        confirmed_at = created_at + timedelta(seconds=random.randint(10, 3600))
                    prediction_row = [
                        current_id,
                        user_id,
                        SERVICE_ID,
                        API_KEY_ID,
                        iris_class,
                        0, # is_deleted
                        MODEL_VERSION,
                        confirmed_class,
                        confirm,
                        created_at.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        confirmed_at.strftime('%Y-%m-%d %H:%M:%S.%f') if confirm else 'NULL',
                        'iris', # type
                    ]
                    prediction_results.append(prediction_row)
                    
                    # 2. IrisResult 상세 데이터 생성
                    sepal_length, sepal_width, petal_length, petal_width = generate_iris_data(iris_class)
                    
                    iris_row = [
                        current_id,
                        sepal_length,
                        sepal_width,
                        petal_length,
                        petal_width,
                        0, # redundancy
                    ]
                    iris_results.append(iris_row)
                    
                    current_id += 1

    print(f"총 {current_id - 1}개의 레코드 생성 완료.")
    print("---------------------------------------")

    # CSV 파일에 쓰기 로직
    with open(PREDICTION_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(prediction_header)
        writer.writerows(prediction_results)
    
    with open(IRIS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(iris_header)
        writer.writerows(iris_results)
        
    print(f"CSV 파일 저장 완료: {PREDICTION_CSV_PATH} 및 {IRIS_CSV_PATH}")

def main():
    # 사용자 데이터 생성 (실제 DB 연결은 upload_csv.py에서 다룸)
    create_users_data()
    
    # CSV 데이터 생성
    generate_csv_data()

if __name__ == '__main__':
    main()
