# apps/loan/cli_upload.py
 
import os
import sys

# 현재 파일 경로 (cli_upload.py)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# 'apps' 디렉터리 (current_dir의 부모)
apps_dir = os.path.abspath(os.path.join(current_dir, '..'))
# 'loan-1' 프로젝트 루트 (apps_dir의 부모)
project_root = os.path.abspath(os.path.join(apps_dir, '..')) 

if project_root not in sys.path:
    # Python이 'apps' 패키지를 찾을 수 있도록 loan-1 디렉터리를 추가
    sys.path.append(project_root)

from apps import create_app  # Flask 앱 팩토리 함수가 apps/__init__.py에 있다고 가정
from apps.loan.db_upload import run_db_upload
from apps.extensions import db # db 객체를 사용하기 위해 임포트

# 1. Flask 애플리케이션 객체 생성
app = create_app()

# 2. 애플리케이션 컨텍스트 활성화
with app.app_context():
    # 3. 데이터베이스 업로드 함수 실행
    run_db_upload()

print("Flask 컨텍스트를 이용한 DB 업로드 완료.")