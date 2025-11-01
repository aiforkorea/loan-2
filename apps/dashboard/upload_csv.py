# apps/dashboard/upload_csv.py
import os
import csv
import sys 
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from datetime import datetime
from flask import Flask
from apps.extensions import db
from apps.dbmodels import User, PredictionResult, UserType, UserType, MatchStatus, generate_password_hash
from apps.iris.dbmodels import IrisResult, IrisClassType

# 현재 파일의 디렉토리
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 (dashboard-2)
PROJECT_ROOT = os.path.join(CURRENT_DIR, '..', '..')
# DB 파일 경로
DB_PATH = os.path.join(PROJECT_ROOT, 'instance', 'mydb.sqlite3')

# CSV 파일 경로
PREDICTION_CSV_PATH = os.path.join(CURRENT_DIR, 'prediction_results.txt')
IRIS_CSV_PATH = os.path.join(CURRENT_DIR, 'iris_results.txt')

def create_app():
    """Flask 애플리케이션 및 DB 설정을 위한 헬퍼 함수"""
    app = Flask(__name__, instance_path=os.path.join(PROJECT_ROOT, 'instance'))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # extensions.py에서 정의된 db 객체를 app에 연결
    db.init_app(app)
    
    # instance 폴더가 없으면 생성
    os.makedirs(os.path.join(PROJECT_ROOT, 'instance'), exist_ok=True)
    
    return app

def upload_users_data(app):
    """사용자 데이터를 DB에 추가하거나 업데이트합니다."""
    # gen_data.py에서 생성된 사용자 데이터 (하드코딩)
    password_input = '1'
    hashed_password = generate_password_hash(password_input)
    users_data = [
        # 전문가 (expert) - id=2
        {'id': 2, 'username': 'kdhong2', 'email': 'kdhong2@example.com', 'password_hash': hashed_password, 'user_type': UserType.EXPERT},
        # 일반 사용자 1 (user) - id=3
        {'id': 3, 'username': 'kdhong3', 'email': 'kdhong3@example.com', 'password_hash': hashed_password, 'user_type': UserType.USER},
        # 일반 사용자 2 (user) - id=4
        {'id': 4, 'username': 'kdhong4', 'email': 'kdhong4@example.com', 'password_hash': hashed_password, 'user_type': UserType.USER},
    ]
    with app.app_context():
        print("--- 사용자 데이터 업로드 시작 ---")
        for data in users_data:
            user = db.session.get(User, data['id'])
            if user is None:
                # 사용자가 없으면 새로 생성
                new_user = User(
                    id=data['id'],
                    username=data['username'],
                    email=data['email'],
                    password_hash=data['password_hash'],
                    user_type=data['user_type'],
                    is_active=True,
                    is_deleted=False,
                    match_status=MatchStatus.UNASSIGNED,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                db.session.add(new_user)
                print(f"새 사용자 추가: ID {new_user.id} ({new_user.email})")
            else:
                # 사용자가 이미 있으면 정보 업데이트 (옵션)
                user.password_hash = data['password_hash']
                user.user_type = data['user_type']
                user.updated_at = datetime.now()
                print(f"기존 사용자 업데이트: ID {user.id} ({user.email})")
        db.session.commit()
        print("사용자 데이터 업로드 완료.")

def load_csv_data(csv_path, headers):
    """CSV 파일 데이터를 로드하여 딕셔너리 리스트로 반환"""
    if not os.path.exists(csv_path):
        print(f"오류: 파일이 존재하지 않습니다. {csv_path}")
        return []
    data_list = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵
        for row in reader:
            data = dict(zip(headers, row))
            # 'NULL' 값 처리 및 타입 변환
            processed_data = {}
            for key, value in data.items():
                if value == 'NULL':
                    processed_data[key] = None
                elif key in ['id', 'user_id', 'service_id', 'api_key_id', 'confirm', 'is_deleted', 'career_years', 'usage_count', 'daily_limit', 'monthly_limit', 'response_status_code']:
                    try:
                        processed_data[key] = int(value)
                    except (ValueError, TypeError):
                        processed_data[key] = None
                elif key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
                    try:
                        processed_data[key] = float(value)
                    except (ValueError, TypeError):
                        processed_data[key] = None
                elif key == 'redundancy':
                    try:
                        processed_data[key] = bool(int(value))
                    except (ValueError, TypeError):
                        processed_data[key] = False
                elif key.endswith('_at') or key.endswith('_date') or key.endswith('_timestamp'):
                    if value:
                        try:
                            processed_data[key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            try:
                                processed_data[key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                processed_data[key] = None
                else:
                    processed_data[key] = value
            data_list.append(processed_data)
    return data_list
def upload_integrated_iris_data(app):
    """prediction_results.txt와 iris_results.txt를 통합하여 IrisResult 레코드들을 생성"""
    print("\n--- 통합 IrisResult 데이터 업로드 시작 ---")
    # CSV 데이터 로드
    prediction_header = [
        "id", "user_id", "service_id", "api_key_id", "predicted_class", "is_deleted",
        "model_version", "confirmed_class", "confirm", "created_at", "confirmed_at", "type"
    ]
    iris_header = [
        "id", "sepal_length", "sepal_width", "petal_length", "petal_width", "redundancy"
    ]
    prediction_data = load_csv_data(PREDICTION_CSV_PATH, prediction_header)
    iris_data = load_csv_data(IRIS_CSV_PATH, iris_header)
    # iris_data를 ID로 인덱싱
    iris_dict = {data['id']: data for data in iris_data}
    with app.app_context():
        uploaded_count = 0
        for pred_data in prediction_data:
            pred_id = pred_data['id']
            iris_info = iris_dict.get(pred_id)
            
            if not iris_info:
                print(f"경고: ID {pred_id}에 해당하는 iris 데이터를 찾을 수 없습니다. 스킵합니다.")
                continue
            
            # 기존 레코드 존재 확인
            existing_record = db.session.get(IrisResult, pred_id)
            if existing_record:
                print(f"경고: ID {pred_id} IrisResult 레코드가 이미 존재합니다. 스킵합니다.")
                continue
            
            # 새로운 IrisResult 레코드 생성 (모든 필드 포함)
            iris_record = IrisResult(
                id=pred_id,
                # PredictionResult 부모 클래스 필드들
                user_id=pred_data['user_id'],
                service_id=pred_data['service_id'],
                api_key_id=pred_data['api_key_id'],
                predicted_class=pred_data['predicted_class'],
                is_deleted=pred_data['is_deleted'],
                model_version=pred_data['model_version'],
                confirmed_class=pred_data['confirmed_class'],
                confirm=pred_data['confirm'],
                created_at=pred_data['created_at'],
                confirmed_at=pred_data['confirmed_at'],
                type='iris',  # 명시적으로 iris 타입 설정
                # IrisResult 고유 필드들
                sepal_length=iris_info['sepal_length'],
                sepal_width=iris_info['sepal_width'],
                petal_length=iris_info['petal_length'],
                petal_width=iris_info['petal_width'],
                redundancy=iris_info['redundancy'],
            )
            
            db.session.add(iris_record)
            uploaded_count += 1
            #print(f"ID {pred_id}: IrisResult 레코드 생성 완료")
        
        # 일괄 커밋
        try:
            db.session.commit()
            print(f"총 {uploaded_count}개의 IrisResult 레코드가 성공적으로 생성되었습니다.")
        except Exception as e:
            db.session.rollback()
            print(f"데이터 업로드 중 오류 발생: {e}")
            raise

def main():
    app = create_app()
    
    with app.app_context():
        # 데이터베이스와 테이블 생성
        db.create_all()

        print("\n--- 기존 데이터 완전 삭제 ---")
        try:
            # 모든 prediction_results 관련 데이터 삭제
            deleted_count = db.session.query(PredictionResult).delete(synchronize_session=False)
            db.session.commit()
            print(f"기존 데이터 삭제 완료: {deleted_count}개 레코드")
            
            # 삭제 확인
            remaining_count = db.session.query(PredictionResult).count()
            print(f"삭제 후 남은 레코드: {remaining_count}개")
            
        except Exception as e:
            db.session.rollback()
            print(f"데이터 삭제 중 오류 발생: {e}")

    # 1. 사용자 데이터 업로드
    upload_users_data(app)
    
    # 2. 통합된 IrisResult 데이터 업로드 (prediction_results + iris_results 결합)
    upload_integrated_iris_data(app)
    
    print("\n--- 모든 데이터 업로드 작업 완료 ---")

if __name__ == '__main__':
    main()