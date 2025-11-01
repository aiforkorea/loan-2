# apps/loan/db_upload.py

import csv
import os
import sys
from datetime import datetime, date
from sqlalchemy.exc import IntegrityError
from flask import current_app # Flask 컨텍스트 내에서 앱 객체에 접근하기 위해 사용

# 상대 경로 임포트를 위해 상위 디렉터리를 sys.path에 추가 (실행 환경에 따라 필요할 수 있음)
# 이 부분은 cli_upload.py에서 처리하므로 주석 처리된 상태를 유지합니다.
# if __name__ == '__main__':
#     # 현재 파일 경로 (apps/loan/db_upload.py) 기준
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # 'apps' 디렉터리의 부모 디렉터리 (프로젝트 루트)를 경로에 추가
#     project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
#     if project_root not in sys.path:
#         sys.path.append(project_root)


# 모델 및 확장 기능 임포트
from apps.extensions import db
from apps.dbmodels import User, UserType
from apps.loan.dbmodels import (
    CustInfo, 
    FamilyInfo, 
    LoanApplicantInfo, 
    LoanDefaultAccount
)

# ----------------- 설정 -----------------
# 현재 파일 (db_upload.py) 기준으로 'legacy' 폴더의 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, 'legacy') 

CUST_INFO_CSV = os.path.join(CSV_DIR, 'mlops.cust_info.csv')
FAMILY_INFO_CSV = os.path.join(CSV_DIR, 'mlops.family_info.csv')
LOAN_APPLICANT_INFO_CSV = os.path.join(CSV_DIR, 'mlops.loan_applicant_info.csv')
LOAN_DEFAULT_ACCOUNT_CSV = os.path.join(CSV_DIR, 'mlops.loan_default_account.csv')

# ----------------- 유틸리티 함수 -----------------

def parse_date(date_str: str) -> date:
    """날짜 문자열(YYYYMMDD)을 datetime.date 객체로 변환"""
    try:
        return datetime.strptime(date_str, '%Y%m%d').date()
    except ValueError as e:
        print(f"Error parsing date string '{date_str}': {e}")
        return None

def convert_to_int(value: str):
    """문자열을 정수로 변환하며, '\\N' 또는 공백 문자열은 None으로 처리"""
    if value in ('\\N', '', None):
        return None
    try:
        return int(value)
    except ValueError:
        # int() 변환 실패 시 경고 출력 및 None 반환 (선택 사항)
        print(f"경고: 정수 변환 실패 값 '{value}'. None으로 처리합니다.")
        return None

def read_unique_cust_ids(file_paths: list) -> set:
    """주어진 파일 경로에서 모든 고유한 cust_id를 추출"""
    unique_ids = set()
    
    csv_configs = [
        (CUST_INFO_CSV, 'cust_id'),
        (FAMILY_INFO_CSV, 'cust_id'),
        (LOAN_APPLICANT_INFO_CSV, 'cust_id'),
    ]

    for file_path, id_column in csv_configs:
        if file_path in file_paths:
            try:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get(id_column):
                            unique_ids.add(row[id_column])
            except FileNotFoundError:
                print(f"경고: 파일 {file_path}를 찾을 수 없습니다. 건너뜁니다.")
            except Exception as e:
                print(f"경고: 파일 {file_path} 처리 중 오류 발생 - {e}")
                
    return unique_ids

# ----------------- 1단계: User 테이블 업로드 및 ID 매핑 -----------------

def upload_users_and_create_map(legacy_cust_ids: set) -> dict:
    """
    고유한 레거시 cust_id를 User 테이블에 삽입하고, 
    {레거시_cust_id: 새_user_id} 형태의 맵을 생성
    """
    user_id_map = {}
    print(f"--- 1단계: User 테이블에 {len(legacy_cust_ids)}명의 사용자 생성 ---")

    for legacy_id in sorted(list(legacy_cust_ids)): # 정렬하여 일관성 유지
        # User 모델에 레거시 ID를 username과 email의 접두사로 사용
        # 단, username은 apps/dbmodels.py에서 unique=True이므로 중복되면 안됨
        
        # 이미 존재하는 사용자인지 확인 (예: 마이그레이션을 여러 번 실행하는 경우)
        existing_user = User.query.filter_by(username=f"legacy_{legacy_id}").first()
        if existing_user:
            user_id_map[legacy_id] = existing_user.id
            # print(f"기존 사용자 매핑: {legacy_id} -> {existing_user.id}")
            continue

        new_user = User(
            username=f"legacy_{legacy_id}",
            email=f"legacy_{legacy_id}@example.com",
            user_type=UserType.USER
        )
        # 비밀번호 해시 설정 (apps/dbmodels.py의 @password.setter 사용)
        new_user.password = 'temp_password_1234'
        
        db.session.add(new_user)
        try:
            db.session.flush() # ID를 얻기 위해 플러시
            user_id_map[legacy_id] = new_user.id
        except IntegrityError as e:
            db.session.rollback()
            print(f"오류: 사용자 {legacy_id} 생성 실패 - {e}")
            continue
    
    db.session.commit()
    print("--- 1단계 완료: User 테이블 생성 및 ID 매핑 완료 ---")
    return user_id_map

# ----------------- 2단계: LoanDefaultAccount 업로드 -----------------

def upload_loan_default_account():
    """LoanDefaultAccount 데이터를 DB에 업로드"""
    print("\n--- 2단계: LoanDefaultAccount 업로드 시작 ---")
    count = 0
    
    try:
        with open(LOAN_DEFAULT_ACCOUNT_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                reg_date = parse_date(row['registration_date'])
                if reg_date is None: continue

                account = LoanDefaultAccount(
                    loan_account_id=row['loan_account_id'],
                    registration_date=reg_date,
                    registration_time=row['registration_time'],
                    loan_default=row['loan_default']
                )
                db.session.add(account)
                count += 1
            
        db.session.commit()
        print(f"--- 2단계 완료: LoanDefaultAccount {count}개 업로드 성공 ---")
    except FileNotFoundError:
        print(f"오류: {LOAN_DEFAULT_ACCOUNT_CSV} 파일을 찾을 수 없습니다.")
    except IntegrityError as e:
        db.session.rollback()
        print(f"오류: LoanDefaultAccount 업로드 실패 - {e}")
    except Exception as e:
        db.session.rollback()
        print(f"예상치 못한 오류: {e}")

# ----------------- 3단계: CustInfo 업로드 -----------------

def upload_cust_info(user_id_map):
    """CustInfo 데이터를 DB에 업로드"""
    print("\n--- 3단계: CustInfo 업로드 시작 ---")
    count = 0
    
    try:
        with open(CUST_INFO_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                legacy_cust_id = row['cust_id']
                new_user_id = user_id_map.get(legacy_cust_id) 
                
                if new_user_id is None:
                    print(f"경고: 레거시 ID {legacy_cust_id}에 매핑된 User ID가 없습니다. 건너뜀.")
                    continue

                base_dt = parse_date(row['base_dt'])
                if base_dt is None: continue

                cust = CustInfo(
                    base_dt=base_dt,
                    cust_id=new_user_id, # 변환된 Integer ID 사용
                    gender=row['gender'],
                    married=row['married'],
                    education=row['education'],
                    self_employed=row['self_employed']
                )
                db.session.add(cust)
                count += 1

        db.session.commit()
        print(f"--- 3단계 완료: CustInfo {count}개 업로드 성공 ---")
    except FileNotFoundError:
        print(f"오류: {CUST_INFO_CSV} 파일을 찾을 수 없습니다.")
    except IntegrityError as e:
        db.session.rollback()
        print(f"오류: CustInfo 업로드 실패 - {e}")
    except Exception as e:
        db.session.rollback()
        print(f"예상치 못한 오류: {e}")
        
# ----------------- 4단계: FamilyInfo 업로드 -----------------

def upload_family_info(user_id_map):
    """FamilyInfo 데이터를 DB에 업로드"""
    print("\n--- 4단계: FamilyInfo 업로드 시작 ---")
    count = 0
    
    try:
        with open(FAMILY_INFO_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                legacy_cust_id = row['cust_id']
                new_user_id = user_id_map.get(legacy_cust_id) 
                
                if new_user_id is None:
                    print(f"경고: 레거시 ID {legacy_cust_id}에 매핑된 User ID가 없습니다. 건너뜀.")
                    continue

                base_dt = parse_date(row['base_dt'])
                if base_dt is None: continue

                family = FamilyInfo(
                    base_dt=base_dt,
                    cust_id=new_user_id, # 변환된 Integer ID 사용
                    family_cust_id=row['family_cust_id'],
                    living_together=row['living_together']
                )
                db.session.add(family)
                count += 1

        db.session.commit()
        print(f"--- 4단계 완료: FamilyInfo {count}개 업로드 성공 ---")
    except FileNotFoundError:
        print(f"오류: {FAMILY_INFO_CSV} 파일을 찾을 수 없습니다.")
    except IntegrityError as e:
        db.session.rollback()
        print(f"오류: FamilyInfo 업로드 실패 - {e}")
    except Exception as e:
        db.session.rollback()
        print(f"예상치 못한 오류: {e}")
        
# ----------------- 5단계: LoanApplicantInfo 업로드 -----------------

def upload_loan_applicant_info(user_id_map):
    """LoanApplicantInfo 데이터를 DB에 업로드"""
    print("\n--- 5단계: LoanApplicantInfo 업로드 시작 ---")
    count = 0
    
    try:
        with open(LOAN_APPLICANT_INFO_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                legacy_cust_id = row['cust_id']
                new_user_id = user_id_map.get(legacy_cust_id)
                
                if new_user_id is None:
                    print(f"경고: 레거시 ID {legacy_cust_id}에 매핑된 User ID가 없습니다. 건너뜀.")
                    continue

                applicant_date = parse_date(row['applicant_date'])
                if applicant_date is None: continue
                
                # BigInteger 필드는 int()로 변환하며, NULL 값(\N) 처리
                # DECIMAL 필드는 float()로 변환하며, NULL 값(\N) 처리
                applicant = LoanApplicantInfo(
                    applicant_id=row['applicant_id'],
                    applicant_date=applicant_date,
                    applicant_time=row['applicant_time'],
                    cust_id=new_user_id, # 변환된 Integer ID 사용
                    
                    # NULL 값 처리 로직 적용 (convert_to_int 함수 사용)
                    applicant_income=convert_to_int(row['applicant_income']),
                    coapplicant_income=convert_to_int(row['coapplicant_income']),
                    
                    # Credit History (float/decimal) NULL 처리
                    credit_history=float(row['credit_history']) if row['credit_history'] not in ('\\N', '') else None,
                    
                    property_area=row['property_area'],
                    
                    # NULL 값 처리 로직 적용 (convert_to_int 함수 사용)
                    loan_amount=convert_to_int(row['loan_amount']),
                    loan_amount_term=convert_to_int(row['loan_amount_term']),
                    
                    loan_account_id=row['loan_account_id']
                )
                db.session.add(applicant)
                count += 1

        db.session.commit()
        print(f"--- 5단계 완료: LoanApplicantInfo {count}개 업로드 성공 ---")
    except FileNotFoundError:
        print(f"오류: {LOAN_APPLICANT_INFO_CSV} 파일을 찾을 수 없습니다.")
    except IntegrityError as e:
        db.session.rollback()
        print(f"오류: LoanApplicantInfo 업로드 실패 - {e}")
    except Exception as e:
        db.session.rollback()
        print(f"예상치 못한 오류: {e}")


# ----------------- 메인 실행 로직 -----------------

def run_db_upload():
    """데이터베이스 업로드 전체 과정을 실행하는 메인 함수"""
    # current_app을 사용하여 Flask 앱 컨텍스트 내에서 실행됨을 가정
    
    print("==============================================")
    print("      CSV Legacy Data Upload Start")
    print("==============================================")

    # 0. 필요한 모든 레거시 cust_id 추출
    all_csv_paths = [CUST_INFO_CSV, FAMILY_INFO_CSV, LOAN_APPLICANT_INFO_CSV]
    legacy_cust_ids = read_unique_cust_ids(all_csv_paths)
    
    if not legacy_cust_ids:
        print("CSV 파일에서 추출된 고유한 cust_id가 없습니다. 경로를 확인하세요.")
        return

    # 1. User 테이블 업로드 및 매핑 생성
    user_id_map = upload_users_and_create_map(legacy_cust_ids)

    if not user_id_map:
        print("User ID 매핑 테이블 생성에 실패했습니다. 다음 단계를 건너뜁니다.")
        return

    # 2. LoanDefaultAccount 업로드 (User ID와 독립적)
    upload_loan_default_account()

    # 3. CustInfo 업로드 (User ID 종속)
    upload_cust_info(user_id_map)

    # 4. FamilyInfo 업로드 (User ID 종속)
    upload_family_info(user_id_map)

    # 5. LoanApplicantInfo 업로드 (User ID 및 LoanDefaultAccount 종속)
    upload_loan_applicant_info(user_id_map)

    print("\n--- 모든 CSV 데이터 업로드 프로세스 완료 ---")
    
# 스크립트를 직접 실행하는 경우 (Flask 쉘/커맨드라인에서 호출됨)
if __name__ == '__main__':
    # 이 부분은 실제 Flask 앱을 생성하고 컨텍스트를 설정하는 
    # 'run.py' 또는 'manage.py' 파일에서 호출되어야 합니다.
    print("이 스크립트는 Flask 애플리케이션 컨텍스트 내에서 'run_db_upload()' 함수를 통해 실행되어야 합니다.")
    print("아래 3번 항목의 실행 방법을 참고하십시오.")