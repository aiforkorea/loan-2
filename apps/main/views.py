# apps/main/views.py
from flask import render_template, current_app, request
#from flask_login import login_required, current_user
from apps.main import main
@main.route("/")
def index():
    # INFO 레벨 로그: 정상적인 동작을 기록
    current_app.logger.info("메인 페이지에 접근했습니다.")
    # DEBUG 레벨 로그: 개발 및 디버깅용 메시지
    # (개발 환경에서만 콘솔에 출력, 배포 환경에서는 기록되지 않음)
    current_app.logger.debug("사용자 IP: %s", request.remote_addr)
    try:
        # 예시: 파일을 열다가 오류가 발생한 상황
        # open('non_existent_file.txt', 'r')
        pass # 현재는 에러가 발생하지 않으므로 pass
    except Exception as e:
        # ERROR 레벨 로그: 예외 발생 시 기록
        current_app.logger.error("파일 처리 중 오류 발생: %s", str(e))
        # WARNING 레벨 로그: 잠재적 문제점을 기록
        current_app.logger.warning("오류 발생으로 인해 특정 기능이 제대로 동작하지 않을 수 있습니다.")
#    if current_user.is_authenticated:
#        return render_template("main/index.html", username=current_user.username)
    return render_template("main/index.html")
@main.route("/services")
def services():
    pass