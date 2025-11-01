# apps/dashboard/__init__.py 예시

from flask import Blueprint

# Blueprint 생성 시 static_folder 인자를 추가해야 합니다.
# -------------------------------------------------------------
# Blueprint 이름: 'dashboard'
# 템플릿 폴더: templates (자동 인식)
# 정적 폴더: static (명시적으로 지정)
dashboard = Blueprint(
    'dashboard',
    __name__,
    template_folder='templates',
    static_folder='static',
    # static_url_path는 Blueprint의 정적 파일을 메인 앱의 정적 파일과 구분하는 URL 경로를 지정합니다.
    # 일반적으로 'static'으로 두는 것이 표준입니다.
    static_url_path='/dashboard_static' 
)

from . import views # views.py에서 정의된 라우트 함수들을 Blueprint에 등록합니다.