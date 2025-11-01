# apps/iris/views.py
import csv, pickle, os, joblib
from io import StringIO
from flask import Flask, Response, flash, redirect, request, render_template, jsonify, abort, current_app, url_for, g
import logging, functools
from sqlalchemy import String, cast, desc, func, or_
from sqlalchemy.orm import joinedload
from apps.extensions import csrf
from apps.dbmodels import PredictionResult, db, APIKey, UsageLog, UsageType, Service, Match, UserType, MatchStatus
from apps.iris.dbmodels import IrisResult, LogStatusType, IrisClassType
import numpy as np
from flask_login import current_user, login_required
from apps.iris.forms import EmptyForm, IrisLogSearchForm, IrisResultConfirmForm, IrisResultEditForm, IrisUserForm
from . import iris
from datetime import datetime, time, timedelta
from apps.decorators import admin_required, expert_required
from apps.config import Config
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
# 1. pickle 이용 업로드
#with open(MODEL_PATH, 'rb') as f:
#    model = pickle.load(f)
# 2. joblib 이용 업로드
model = joblib.load(MODEL_PATH)
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']
@iris.route('/services')
@login_required
def services():
    current_app.logger.debug("search_query: %s", "Starts services")
    return render_template('iris/services.html')
@iris.route('/iris_predict', methods=['GET', 'POST'])
@login_required
def iris_predict():
    iris_service_id = 1    
    form = IrisUserForm()
    if form.validate_on_submit():
        sepal_length = form.sepal_length.data
        sepal_width = form.sepal_width.data
        petal_length = form.petal_length.data
        petal_width = form.petal_width.data
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        existing_result = IrisResult.query.filter_by(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            is_deleted=False, # Soft delete된 데이터는 제외
            user_id=current_user.id # 전체 DB에서 중복확인시, 라인 삭제
        ).first()
        can_confirm = current_user.is_expert() or current_user.is_admin()
        existing_result_found = False
        # 입력 데이터 딕셔너리 생성
        input_data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        if existing_result:    # 중복이면
            existing_result_found = True
            # 확인 클래스 값이 None이면 '미확인 not available'으로 표시
            confirmed_class_display = existing_result.confirmed_class if existing_result.confirmed_class else "not available"
            # log 추가
            redundancy_log = UsageLog(
                user_id=current_user.id,
                service_id=iris_service_id,
                usage_type=UsageType.WEB_UI,
                endpoint=request.path,
                inference_timestamp=datetime.now(),
                remote_addr=request.remote_addr,
                response_status_code=200,
                request_data_summary=str(input_data)[:200],
                log_status='중복',
                prediction_result_id=existing_result.id
            )
            db.session.add(redundancy_log)
            db.session.commit()
            return render_template('iris/predict.html',
                                   result=existing_result.predicted_class,
                                   confirmed_class=confirmed_class_display,
                                   sepal_length=sepal_length,
                                   sepal_width=sepal_width,
                                   petal_length=petal_length,
                                   petal_width=petal_width,
                                   form=form,
                                   TARGET_NAMES=TARGET_NAMES,
                                   iris_result_id=existing_result.id,
                                   allow_confirm_save=False,
                                   existing_result_found=existing_result_found)
        else:               # 새로운 입력값인 경우
            pred = model.predict(features)[0]
            new_iris_result = IrisResult(
                user_id=current_user.id,
                service_id=iris_service_id,  
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
                predicted_class=TARGET_NAMES[pred],  
                model_version='1.0',  
                confirm=False  
            )
            db.session.add(new_iris_result)
            db.session.flush()
            new_usage_log = UsageLog(
                user_id=current_user.id,
                usage_type=UsageType.WEB_UI,
                endpoint=request.path,
                remote_addr=request.remote_addr,
                response_status_code=200,
                inference_timestamp=datetime.now(),
                service_id=iris_service_id,
                request_data_summary=str(input_data)[:200],
                prediction_result_id=new_iris_result.id
            )
            db.session.add(new_usage_log)
            db.session.commit()
            iris_result_id = new_iris_result.id
            return render_template('iris/predict.html',
                                 result=TARGET_NAMES[pred],
                                 confirmed_class="not available",
                                 sepal_length=sepal_length, sepal_width=sepal_width,
                                 petal_length=petal_length, petal_width=petal_width, form=form,
                                 TARGET_NAMES=TARGET_NAMES, iris_result_id=iris_result_id,
                                 allow_confirm_save=can_confirm,
                                 existing_result_found=existing_result_found)
    return render_template('iris/predict.html', form=form, existing_result_found=False, confirmed_class="not available")
@iris.route('/save_iris_data', methods=['POST'])
@login_required
def save_iris_data():
    if request.method != 'POST':
        flash('잘못된 접근입니다.', 'danger')
        return redirect(url_for('iris.iris_predict'))
    result_id = request.form.get('iris_result_id')
    confirmed_class = request.form.get('confirmed_class')
    # 전문가와 관리자만 이 함수에 접근하도록 권한 확인을 추가 (보안 강화)
    if not (current_user.is_expert() or current_user.is_admin()):
        flash('추론 결과를 확인 저장할 권한이 없습니다.', 'danger')
        abort(403)
    if not result_id or confirmed_class not in ['setosa', 'versicolor', 'virginica']:
        flash('유효한 데이터 ID 또는 품종이 아닙니다.', 'danger')
        return redirect(url_for('iris.iris_predict'))
    try:
        result = IrisResult.query.filter_by(id=result_id).first_or_404()
        # 사용자가 자신의 결과가 아닌 다른 사람의 결과를 수정하려는 경우도 방지
        if result.user_id != current_user.id and not current_user.is_admin() and not current_user.is_expert():
            abort(403)
        # 이미 확인된 결과는 다시 저장할 수 없도록 로직 추가
        if result.confirm:
            flash("이미 확인된 결과는 다시 저장할 수 없습니다.", 'warning')
            return redirect(url_for('iris.iris_predict'))
        result.confirmed_class = confirmed_class
        result.confirm = True
        result.confirmed_at = datetime.now()
        recent_log = UsageLog.query.filter_by(prediction_result_id=result.id).order_by(desc(UsageLog.timestamp)).first()
        new_usage_log = UsageLog(
            user_id=current_user.id,
            service_id=recent_log.service_id if recent_log else None,
            api_key_id=recent_log.api_key_id if recent_log else None,
            endpoint=request.path,
            usage_type=UsageType.WEB_UI,
            log_status='추론확인',
            inference_timestamp=recent_log.inference_timestamp if recent_log else None,
            remote_addr=request.remote_addr,
            response_status_code=200,
            prediction_result_id=result.id
        )
        db.session.add(new_usage_log)
        db.session.commit()
        flash('추론 확인 및 관련 로그가 성공적으로 처리되었습니다.', 'success')
        return redirect(url_for('iris.iris_predict'))
    except Exception as e:
        db.session.rollback()
        flash(f'결과 입력 중 오류가 발생했습니다: {e}', 'danger')
        return redirect(url_for('iris.iris_predict'))
@iris.route('/results')
@login_required
def results():
    search_query = request.args.get('search', '', type=str)
    confirm_query = request.args.get('confirm', '', type=str)
    date_filter_type = request.args.get('date_filter_type', '', type=str)
    start_date_str = request.args.get('start_date', '', type=str)
    end_date_str = request.args.get('end_date', '', type=str)
    page = request.args.get('page', 1, type=int)
    per_page = 10
    # 쿼리 기본 설정
    query = IrisResult.query
    # CSRF 토큰을 처리할 폼 객체 생성, 템플릿으로 이 폼들을 전달하여 각 폼에 CSRF 토큰이 포함되도록 합니다.
    confirm_form = IrisResultConfirmForm()
    edit_form = IrisResultEditForm()
    delete_form = EmptyForm()
    # 사용자 권한에 따른 필터링 로직
    if current_user.is_admin():
        pass # 관리자는 모든 결과를 볼 수 있음
    elif current_user.is_expert():
        # 전문가와 매칭된 사용자 ID 목록 가져오기
        matched_user_ids = [m.user_id for m in Match.query.filter_by(expert_id=current_user.id, status=MatchStatus.IN_PROGRESS).all()]
        query = query.filter(
            or_(
                IrisResult.user_id.in_(matched_user_ids),
                IrisResult.user_id == current_user.id
            )
        )
    else: # 일반 사용자
        query = query.filter_by(user_id=current_user.id)
    has_date_filter_error = False
    if start_date_str or end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else None
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None
            if start_date and end_date and start_date > end_date:
                flash('시작일은 종료일보다 이전이어야 합니다.', 'danger')
                has_date_filter_error = True
            else:
                filter_col = IrisResult.confirmed_at if date_filter_type == 'confirmed_at' else IrisResult.created_at
                if start_date and end_date:
                    next_day = end_date + timedelta(days=1)
                    query = query.filter(filter_col >= start_date, filter_col < next_day)
                elif start_date:
                    query = query.filter(filter_col >= start_date)
                elif end_date:
                    next_day = end_date + timedelta(days=1)
                    query = query.filter(filter_col < next_day)
        except ValueError:
            flash('날짜 입력 또는 기준일자 선택이 잘못되었습니다.', 'danger')
            has_date_filter_error = True
    if search_query:
        query = query.filter(
            or_(
                IrisResult.predicted_class.ilike(f'%{search_query}%'),
                IrisResult.confirmed_class.ilike(f'%{search_query}%'),
                cast(IrisResult.id, String).ilike(f'%{search_query}%'),
            )
        )
    if confirm_query:
        if confirm_query == 'true':
            query = query.filter(IrisResult.confirmed_class.isnot(None))
        elif confirm_query == 'false':
            query = query.filter(IrisResult.confirmed_class.is_(None))
    query = query.filter(IrisResult.is_deleted == False)
    query = query.order_by(IrisResult.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    user_results = pagination.items
    form = EmptyForm()
    filtered_args = {
        'search': search_query,
        'confirm': confirm_query,
        'date_filter_type': date_filter_type,
        'start_date': start_date_str,
        'end_date': end_date_str
    }
    return render_template(
        'iris/user_results.html',
        title='추론결과',
        results=user_results,
        pagination=pagination,
        filtered_args=filtered_args,
        search_query=search_query,
        confirm_query=confirm_query,
        date_filter_type=date_filter_type,
        start_date=start_date_str,
        end_date=end_date_str,
        # CSRF 토큰 처리를 위해 폼 객체들을 템플릿에 추가로 전달합니다.
        confirm_form=confirm_form,
        edit_form=edit_form,
        delete_form=delete_form
    )
@iris.route('/results/<int:result_id>/confirm', methods=['POST'])
@login_required
def confirm_result(result_id):
    # 요청을 처리하기 전에 폼 객체를 생성하고 유효성을 검사합니다.
    # form.validate_on_submit()은 POST 요청이고 CSRF 토큰이 유효한지 확인합니다.
    form = IrisResultConfirmForm()
    if form.validate_on_submit():
        result = IrisResult.query.get_or_404(result_id)

        if not (current_user.id == result.user_id or current_user.is_admin() or (current_user.is_expert() and current_user.id in [m.expert_id for m in Match.query.filter_by(user_id=result.user_id, status=MatchStatus.IN_PROGRESS).all()])):
            flash("이 결과를 확인할 권한이 없습니다.", "danger")
            return redirect(url_for('iris.results'))

        if result.confirmed_class:
            flash('이미 확인된 추론 결과입니다.', 'warning')
            return redirect(url_for('iris.results'))

        result.confirmed_class = form.confirmed_class.data
        result.confirmed_at = datetime.now()
        result.confirm = True
        db.session.commit()
        log_entry = UsageLog(
            user_id=current_user.id,
            service_id=result.service_id,
            prediction_result_id=result.id,
            usage_type=UsageType.WEB_UI,
            log_status=LogStatusType.CONFIRMED.value,
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200
        )
        db.session.add(log_entry)
        db.session.commit()
        flash('추론 결과가 성공적으로 확인되었습니다.', 'success')
    else:
        # 폼 유효성 검사 실패 시
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{field.capitalize()}: {error}", "danger")
    return redirect(url_for('iris.results'))
@iris.route('/results/<int:result_id>/edit_confirmed_class', methods=['POST'])
@login_required
def edit_confirmed_class(result_id):
    form = IrisResultEditForm()
    if form.validate_on_submit():
        result = IrisResult.query.get_or_404(result_id)

        if not (current_user.id == result.user_id or current_user.is_admin() or (current_user.is_expert() and current_user.id in [m.expert_id for m in Match.query.filter_by(user_id=result.user_id, status=MatchStatus.IN_PROGRESS).all()])):
            flash("이 결과를 수정할 권한이 없습니다.", "danger")
            return redirect(url_for('iris.results'))

        if not result.confirmed_class:
            flash("아직 확인되지 않은 결과는 수정할 수 없습니다.", "warning")
            return redirect(url_for('iris.results'))
        
        result.confirmed_class = form.confirmed_class.data
        db.session.commit()
        log_entry = UsageLog(
            user_id=current_user.id,
            service_id=result.service_id,
            prediction_result_id=result.id,
            usage_type=UsageType.WEB_UI,
            log_status=LogStatusType.CONFIRMED_EDITED.value,
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200
        )
        db.session.add(log_entry)
        db.session.commit()

        flash('추론 결과가 성공적으로 수정되었습니다.', 'success')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{field.capitalize()}: {error}", "danger")
    return redirect(url_for('iris.results'))
@iris.route('/results/<int:result_id>/delete', methods=['POST'])
@login_required
def delete_result(result_id):
    form = EmptyForm()
    if form.validate_on_submit():
        result = IrisResult.query.get_or_404(result_id)
        if not (current_user.id == result.user_id or current_user.is_admin() or (current_user.is_expert() and current_user.id in [m.expert_id for m in Match.query.filter_by(user_id=result.user_id, status=MatchStatus.IN_PROGRESS).all()])):
            flash("이 결과를 삭제할 권한이 없습니다.", "danger")
            return redirect(url_for('iris.results'))
        result.is_deleted = True
        db.session.commit()
        log_entry = UsageLog(
            user_id=current_user.id,
            service_id=result.service_id,
            prediction_result_id=result.id,
            usage_type=UsageType.WEB_UI,
            log_status=LogStatusType.DELETED.value,
            endpoint=request.path,
            remote_addr=request.remote_addr,
            response_status_code=200
        )
        db.session.add(log_entry)
        db.session.commit()
        flash('추론 결과가 삭제되었습니다.', 'success')
    return redirect(url_for('iris.results'))
@iris.route('/results/download-csv')
@login_required
def results_download_csv():
    if not (current_user.is_admin() or current_user.is_expert() or current_user.is_user()):
        flash("이 기능에 접근할 권한이 없습니다.", "danger")
        return redirect(url_for('iris.results'))
    query = IrisResult.query
    if current_user.is_admin():
        pass
    elif current_user.is_expert():
        matched_user_ids = [m.user_id for m in Match.query.filter_by(expert_id=current_user.id, status=MatchStatus.IN_PROGRESS).all()]
        query = query.filter(
            or_(
                IrisResult.user_id.in_(matched_user_ids),
                IrisResult.user_id == current_user.id
            )
        )
    else:
        query = query.filter_by(user_id=current_user.id)
    search_query = request.args.get('search', '')
    if search_query:
        keyword = f"%{search_query}%"
        query = query.filter(
            or_(
                IrisResult.predicted_class.ilike(keyword),
                IrisResult.confirmed_class.ilike(keyword),
                cast(IrisResult.id, String).ilike(keyword)
            )
        )
    confirm_query = request.args.get('confirm', '')
    if confirm_query == 'true':
        query = query.filter(IrisResult.confirmed_class.isnot(None))
    elif confirm_query == 'false':
        query = query.filter(IrisResult.confirmed_class.is_(None))
    date_filter_type = request.args.get('date_filter_type', '')
    start_date_str = request.args.get('start_date', '')
    end_date_str = request.args.get('end_date', '')
    if start_date_str or end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
            if end_date:
                end_of_day = datetime.combine(end_date, time.max)
            else:
                end_of_day = None
            filter_col = getattr(IrisResult, date_filter_type) if date_filter_type else IrisResult.created_at
            if start_date and end_of_day:
                query = query.filter(filter_col.between(start_date, end_of_day))
            elif start_date:
                query = query.filter(filter_col >= start_date)
            elif end_of_day:
                query = query.filter(filter_col <= end_of_day)
        except (ValueError, AttributeError):
            flash("유효하지 않은 날짜 형식 또는 기준일자입니다.", "danger")
            return redirect(url_for('iris.results'))
    results = query.order_by(IrisResult.created_at.desc()).all()
    output = StringIO()
    writer = csv.writer(output)
    headers = [
        "ID", "사용자ID", "꽃받침길이", "꽃받침너비", "꽃잎길이", "꽃잎너비",
        "예측품종", "확인품종", "추론시간", "확인시간"
    ]
    writer.writerow(headers)
    for result in results:
        row = [
            result.id,
            result.user_id,
            result.sepal_length,
            result.sepal_width,
            result.petal_length,
            result.petal_width,
            result.predicted_class,
            result.confirmed_class if result.confirmed_class else '미확인',
            result.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            result.confirmed_at.strftime('%Y-%m-%d %H:%M:%S') if result.confirmed_at else '-'
        ]
        writer.writerow(row)
    output_str = output.getvalue()
    output_bytes = output_str.encode('utf-8-sig')
    output.close()
    response = Response(output_bytes, mimetype='text/csv; charset=utf-8-sig')
    filename = f'iris_result_results_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response
@iris.route('/logs', methods=['GET', 'POST'])
@login_required
def logs():
    per_page = 10
    form = IrisLogSearchForm(request.form if request.method == 'POST' else request.args)
    filtered_args = {}

    logs_query = UsageLog.query.options(
        joinedload(UsageLog.user),
        joinedload(UsageLog.api_key),
        joinedload(UsageLog.service)
    )
    # 사용자 권한에 따른 로그 조회 범위 설정
    # 1. 현재는 id를 기준으로 판단, admin 처리내용을 expert가 볼 수 없음, 볼수 있는 방법은? 항목으로 처리하면? 
    if current_user.is_admin():
        # 관리자는 모든 로그를 볼 수 있음
        pass
    elif current_user.is_expert():
        matched_user_ids = [m.user_id for m in Match.query.filter_by(expert_id=current_user.id, status=MatchStatus.IN_PROGRESS).all()]
        logs_query = logs_query.filter(
            or_(
                UsageLog.user_id.in_(matched_user_ids), 
                UsageLog.user_id == current_user.id,
                # 2 (위1번질의해결방안)신규조건추가: log의 prediction_result_id를 통해 원본추론결과의user_id를 확인
                UsageLog.prediction_result.has(IrisResult.user_id.in_(matched_user_ids))
            )
        )
    else:  # 일반 사용자
        logs_query = logs_query.filter_by(user_id=current_user.id)
    # 폼 데이터에 따라 쿼리 필터링
    if form.validate() or request.method == 'GET':
        if form.keyword.data:
            keyword = f"%{form.keyword.data}%"
            logs_query = logs_query.filter(
                or_(
                    cast(UsageLog.user_id, String).ilike(keyword),
                    cast(UsageLog.service_id, String).ilike(keyword),
                    cast(UsageLog.prediction_result_id, String).ilike(keyword),
                )
            )
            filtered_args['keyword'] = form.keyword.data
        if form.usage_type.data:
            logs_query = logs_query.filter(UsageLog.usage_type.ilike(form.usage_type.data))
            filtered_args['usage_type'] = form.usage_type.data
        if form.log_status.data:
            logs_query = logs_query.filter(UsageLog.log_status.ilike(form.log_status.data))
            filtered_args['log_status'] = form.log_status.data
        # 날짜 필터링
        if form.start_date.data and form.end_date.data:
            start_of_day = datetime.combine(form.start_date.data, time.min)
            end_of_day = datetime.combine(form.end_date.data, time.max)
            date_field = getattr(UsageLog, form.date_field.data)
            logs_query = logs_query.filter(date_field.between(start_of_day, end_of_day))
            filtered_args['start_date'] = form.start_date.data.isoformat()
            filtered_args['end_date'] = form.end_date.data.isoformat()
            filtered_args['date_field'] = form.date_field.data
    if request.method == 'POST':
        return redirect(url_for('iris.logs', **filtered_args))
    page = request.args.get('page', 1, type=int)
    logs_pagination = logs_query.order_by(UsageLog.timestamp.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    return render_template(
        'iris/user_logs.html',
        title='AI 로그 이력',
        form=form,
        logs=logs_pagination.items,
        pagination=logs_pagination,
        filtered_args=filtered_args,
    )
@iris.route('/logs/download-csv')
@login_required
def logs_download_csv():
    form = IrisLogSearchForm(request.args)
    logs_query = UsageLog.query.options(
        joinedload(UsageLog.user),
        joinedload(UsageLog.api_key),
        joinedload(UsageLog.service)
    )
    # 사용자 권한에 따른 로그 조회 범위 설정 (위 logs() 함수와 동일하게 적용)
    if current_user.is_admin():
        pass
    elif current_user.is_expert():
        matched_user_ids = [m.user_id for m in Match.query.filter_by(expert_id=current_user.id, status=MatchStatus.IN_PROGRESS).all()]
        logs_query = logs_query.filter(
            or_(
                UsageLog.user_id.in_(matched_user_ids), 
                UsageLog.user_id == current_user.id,
                UsageLog.prediction_result.has(IrisResult.user_id.in_(matched_user_ids))  # 추가 부분
            )
        )
    else:
        logs_query = logs_query.filter_by(user_id=current_user.id)
    # 쿼리 파라미터에 따라 필터링 (logs() 함수와 동일한 로직)
    if form.keyword.data:
        keyword = f"%{form.keyword.data}%"
        logs_query = logs_query.filter(
            or_(
                cast(UsageLog.user_id, String).ilike(keyword),
                cast(UsageLog.service_id, String).ilike(keyword),
                cast(UsageLog.prediction_result_id, String).ilike(keyword),
            )
        )
    if form.usage_type.data:
        logs_query = logs_query.filter(UsageLog.usage_type.ilike(form.usage_type.data))
    if form.log_status.data:
        logs_query = logs_query.filter(UsageLog.log_status.ilike(form.log_status.data))
    if form.start_date.data and form.end_date.data:
        start_of_day = datetime.combine(form.start_date.data, time.min)
        end_of_day = datetime.combine(form.end_date.data, time.max)
        date_field = getattr(UsageLog, form.date_field.data)
        logs_query = logs_query.filter(date_field.between(start_of_day, end_of_day))
    logs = logs_query.order_by(UsageLog.timestamp.desc()).all()
    output = StringIO()
    writer = csv.writer(output)
    headers = [
        "ID", "사용자 ID", "서비스 ID", "추론 ID", "로그 타입", "로그 상태",
        "엔드포인트", "추론 시각", "로그 시각", "원격 주소", "응답 상태 코드"
    ]
    writer.writerow(headers)
    for log in logs:
        row = [
            log.id, log.user_id, log.service_id, log.prediction_result_id,
            log.usage_type.value, log.log_status,
            log.endpoint,
            log.inference_timestamp.strftime('%Y-%m-%d %H:%M:%S') if log.inference_timestamp else '-',
            log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            log.remote_addr, log.response_status_code
        ]
        writer.writerow(row)
    output_str = output.getvalue()
    output_bytes = output_str.encode('utf-8-sig')
    output.close()
    response = Response(output_bytes, mimetype='text/csv; charset=utf-8-sig')
    response.headers['Content-Disposition'] = f'attachment; filename=iris_log_results_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    return response
@iris.route('/api/predict', methods=['POST'])
@csrf.exempt
def api_predict():
    current_app.logger.info("API predict request received.")
    auth_header = request.headers.get('X-API-Key')
    if not auth_header:
        return jsonify({"error": "API Key is required"}), 401
    api_key_entry = APIKey.query.filter_by(key_string=auth_header, is_active=True).first()
    if not api_key_entry:
        return jsonify({"error": "Invalid or inactive API Key"}), 401
    if model is None:
        logging.error("Model is not loaded. Cannot process prediction.")
        return jsonify({"error": "Service is temporarily unavailable."}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
    except ValueError:
        return jsonify({"error": "Invalid data type for Iris features. All fields must be numbers."}), 400
    try:
        iris_service_id = 1
        existing_result = IrisResult.query.filter_by(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            is_deleted=False, # Soft delete된 데이터는 제외
            user_id=api_key_entry.user_id # 전체 DB에서 중복확인시 라인 삭제
        ).first()
        if existing_result:
            confirmed_class_display = existing_result.confirmed_class if existing_result.confirmed_class else "not available"
            redundancy_log = UsageLog(
                user_id=api_key_entry.user_id,
                service_id=iris_service_id,
                api_key_id=api_key_entry.id,
                usage_type=UsageType.API_KEY,
                endpoint=request.path,
                inference_timestamp=datetime.now(),
                remote_addr=request.remote_addr,
                response_status_code=200,
                request_data_summary=str(data)[:200],
                log_status='중복',
                prediction_result_id=existing_result.id
            )
            db.session.add(redundancy_log)
            db.session.commit()
            return jsonify({
                "message": "This prediction already exists in your history.",
                "predicted_class": existing_result.predicted_class,
                "confirmed_class": confirmed_class_display,
                "created_at": existing_result.created_at.isoformat() if existing_result.created_at else None,
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }), 200
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_index = model.predict(features)[0]
        predicted_class_name = TARGET_NAMES[pred_index]
        new_iris_entry = IrisResult(
            user_id=api_key_entry.user_id,
            service_id=iris_service_id,
            api_key_id=api_key_entry.id,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            predicted_class=predicted_class_name,
            model_version='1.0',
            confirm=False,
            is_deleted=False
        )
        db.session.add(new_iris_entry)
        db.session.flush()
        new_usage_log = UsageLog(
            user_id=api_key_entry.user_id,
            service_id=iris_service_id,
            api_key_id=api_key_entry.id,
            usage_type=UsageType.API_KEY,
            endpoint=request.path,
            inference_timestamp=datetime.now(),
            remote_addr=request.remote_addr,
            response_status_code=200,
            request_data_summary=str(data)[:200],
            log_status='정상',
            prediction_result_id=new_iris_entry.id
        )
        db.session.add(new_usage_log)
        db.session.commit()
        return jsonify({
            "predicted_class": predicted_class_name,
            "confirmed_class": "not available",
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error processing API predict request: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

"""
윈도우 CMD
curl -X POST "http://localhost:5000/iris/api/predict" -H "Content-Type: application/json" -H "X-API-Key: your_api_key" -d "{\"sepal_length\":6.0,\"sepal_width\":3.5,\"petal_length\":4.5,\"petal_width\":1.5}"
윈도우 파워쉘
$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = "your_api_key"
}
$body = @{
    sepal_length = 6.0
    sepal_width = 3.5
    petal_length = 4.5
    petal_width = 1.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/iris/api/predict" -Method Post -Headers $headers -Body $body

"""
