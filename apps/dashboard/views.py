# apps/dashboard/views.py
import csv
import joblib
import pandas as pd
import numpy as np
from scipy.stats import kstest

from io import StringIO
from flask import Flask, Response, flash, redirect, request, render_template, jsonify, abort, current_app, url_for, g
import pickle, os
import logging, functools
from sklearn.calibration import label_binarize
from sqlalchemy import String, cast, desc, func, or_
from sqlalchemy.orm import joinedload
from apps.extensions import csrf
from apps.dbmodels import PredictionResult, db, APIKey, UsageLog, UsageType, Service, Match, UserType, MatchStatus
from apps.iris.dbmodels import IrisResult, LogStatusType, IrisClassType
import numpy as np
from flask_login import current_user, login_required
from datetime import datetime, time, timedelta
from apps.decorators import admin_required, expert_required
from apps.config import Config

#from apps import dashboard
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from . import dashboard # 순환 참조가 발생하지 않는다면 이렇게 사용

#MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
#with open(MODEL_PATH, 'rb') as f:
#    model = pickle.load(f)
TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

# MLOps 대시보드 메인 페이지
@dashboard.route('/')
@login_required
@admin_required
def index():
    return render_template('dashboard/index.html')

# AI 성능 지표 및 시각화 데이터 API
@dashboard.route('/api/model_performance')
@login_required
@admin_required
def model_performance():
    # 전문가가 확인(confirm)한 실제 데이터만 사용
    results = IrisResult.query.filter_by(confirm=True).filter_by(is_deleted=False).all()

    if not results:
        return jsonify({'metrics': {}, 'roc_pr_data': {}, 'confusion_matrix': []})

    true_labels = [r.confirmed_class for r in results]
    predicted_labels = [r.predicted_class for r in results]

    # 클래스 매핑
    target_names = ['setosa', 'versicolor', 'virginica']
    true_labels_int = [target_names.index(l) for l in true_labels]
    predicted_labels_int = [target_names.index(l) for l in predicted_labels]

    accuracy = accuracy_score(true_labels_int, predicted_labels_int)
    precision = precision_score(true_labels_int, predicted_labels_int, average='weighted', zero_division=0)
    recall = recall_score(true_labels_int, predicted_labels_int, average='weighted', zero_division=0)
    f1 = f1_score(true_labels_int, predicted_labels_int, average='weighted', zero_division=0)
    
    # ✨ 혼동 행렬 계산
    cm = confusion_matrix(true_labels_int, predicted_labels_int, labels=range(len(target_names)))
    cm_list = cm.tolist() # JSON 직렬화를 위해 리스트로 변환

    # ROC/PR Curve 데이터 (다중 클래스)
    try:
        # 모델을 다시 로드하여 확률값 계산
        MODEL_PATH = 'apps/iris/model.pkl'
        print(MODEL_PATH)
        iris_model = joblib.load(MODEL_PATH)
        input_data = np.array([[r.sepal_length, r.sepal_width, r.petal_length, r.petal_width] for r in results])
        probas = iris_model.predict_proba(input_data)
        
        # 실제 레이블을 이진화하여 다중 클래스 AUC 계산 준비
        true_labels_bin = label_binarize(true_labels_int, classes=[0, 1, 2])
        
        # ROC Curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, name in enumerate(target_names):
            fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], probas[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # PR Curve
        precision_curve = dict()
        recall_curve = dict()
        pr_auc = dict()
        for i, name in enumerate(target_names):
            precision_curve[i], recall_curve[i], _ = precision_recall_curve(true_labels_bin[:, i], probas[:, i])
            pr_auc[i] = auc(recall_curve[i], precision_curve[i])

        # 마이크로, 매크로, 가중치 평균 AUC 계산
        # 마이크로 평균: 모든 클래스의 FPR, TPR을 합쳐서 계산
        fpr_micro, tpr_micro, _ = roc_curve(true_labels_bin.ravel(), probas.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        # 매크로 평균: 각 클래스별 AUC의 단순 평균
        roc_auc_macro = np.mean(list(roc_auc.values()))

        # 가중치 평균: 각 클래스별 AUC에 클래스 샘플 수로 가중치 부여
        class_counts = [true_labels_int.count(i) for i in range(len(target_names))]
        total_count = len(true_labels_int)
        roc_auc_weighted = np.average(list(roc_auc.values()), weights=class_counts)
        
        # 데이터 반환을 위한 JSON 직렬화
        roc_data = {
            'fpr': {name: fpr[i].tolist() for i, name in enumerate(target_names)},
            'tpr': {name: tpr[i].tolist() for i, name in enumerate(target_names)},
            'auc': {name: roc_auc[i] for i, name in enumerate(target_names)},
            'auc_micro': roc_auc_micro,
            'auc_macro': roc_auc_macro,
            'auc_weighted': roc_auc_weighted,
        }
        
        pr_data = {
            'precision': {name: precision_curve[i].tolist() for i, name in enumerate(target_names)},
            'recall': {name: recall_curve[i].tolist() for i, name in enumerate(target_names)},
            'auc': {name: pr_auc[i] for i, name in enumerate(target_names)}
        }
        
    except Exception as e:
        print(f"Error calculating ROC/PR curves: {e}")
        probas = None
        roc_data = {}
        pr_data = {}
        # 오류 발생 시 평균값도 0 또는 None으로 초기화
        roc_auc_micro = None
        roc_auc_macro = None
        roc_auc_weighted = None
    
    # auc_roc와 pr_auc 값을 metrics에 추가
    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_roc_by_class': {target_names[i]: round(roc_auc[i], 4) for i in roc_auc} if roc_auc else {},
        'pr_auc_by_class': {target_names[i]: round(pr_auc[i], 4) for i in pr_auc} if pr_auc else {},
        'auc_roc_micro': round(roc_auc_micro, 4) if roc_auc_micro else None,
        'auc_roc_macro': round(roc_auc_macro, 4) if roc_auc_macro else None,
        'auc_roc_weighted': round(roc_auc_weighted, 4) if roc_auc_weighted else None,
    }

    return jsonify({
        'metrics': metrics,
        'roc_data': roc_data,
        'pr_data': pr_data,
        'confusion_matrix': {
            'matrix': cm_list,
            'labels': target_names
        }
    })
# 데이터 드리프트 감지 및 시각화 API
@dashboard.route('/api/data_drift')
@login_required
@admin_required
def data_drift():
    # 필요한 라이브러리 import (예: datetime, timedelta, pd, kstest, jsonify)가 가정됨
    from datetime import datetime, timedelta
    import pandas as pd
    from scipy.stats import kstest
    from flask import jsonify
    
    # IrisResult 모델은 정의되어 있다고 가정합니다.
    # 예: IrisResult.query.filter(...)

    # 기준 데이터 (과거 30일) vs 최근 데이터 (최근 7일)
    cutoff_date_ref = datetime.now() - timedelta(days=30)
    cutoff_date_curr = datetime.now() - timedelta(days=7)

    # 실제 데이터베이스 쿼리 (모델 및 ORM 설정에 따라 달라짐)
    # 현재는 mock 데이터 또는 실제 쿼리 결과가 있다고 가정
    try:
        # 실제 환경에서는 아래와 같이 데이터베이스에서 데이터를 가져와야 합니다.
        # 예시를 위해 쿼리 로직은 그대로 유지합니다.
        # from apps.models import IrisResult # IrisResult 모델 임포트가 필요할 수 있습니다.
        
        # NOTE: IrisResult 쿼리 부분은 실행 환경에 따라 정확히 정의되어 있어야 합니다.
        # 데이터가 없을 경우를 대비해 빈 리스트로 처리합니다.
        reference_results = IrisResult.query.filter(
            IrisResult.created_at >= cutoff_date_ref,
            IrisResult.created_at < cutoff_date_curr
        ).all()
        current_results = IrisResult.query.filter(
            IrisResult.created_at >= cutoff_date_curr
        ).all()
        
    except Exception as e:
        # DB 연결 실패 등 예외 처리
        print(f"Database query error: {e}")
        return jsonify({'drift_results': {}, 'drift_data': {}})

    if not reference_results or not current_results:
        # 데이터가 부족한 경우
        return jsonify({'drift_results': {}, 'drift_data': {}})
    ref_df = pd.DataFrame([
        {'sepal_length': r.sepal_length, 'sepal_width': r.sepal_width, 'petal_length': r.petal_length, 'petal_width': r.petal_width}
        for r in reference_results
    ])
    curr_df = pd.DataFrame([
        {'sepal_length': r.sepal_length, 'sepal_width': r.sepal_width, 'petal_length': r.petal_length, 'petal_width': r.petal_width}
        for r in current_results
    ])
    drift_results = {}
    drift_data = {}
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for feature in features:
        if ref_df[feature].empty or curr_df[feature].empty:
            continue
        # Kolmogorov-Smirnov 검정
        statistic, p_value = kstest(ref_df[feature], curr_df[feature])
        drift_detected_bool = p_value < 0.05
        # JSON 직렬화 오류를 피하기 위해 Boolean을 정수(1 또는 0)로 변환
        drift_detected_int = 1 if drift_detected_bool else 0
        drift_results[feature] = {
            'p_value': round(p_value, 4),
            # 정수형으로 전달 (0: 정상, 1: 감지됨)
            'drift_detected': drift_detected_int, 
            'message': '드리프트 감지!' if drift_detected_bool else '정상'
        }
        # Plotly.js 히스토그램을 위해 원시 데이터 리스트만 전달
        drift_data[feature] = {
            'reference': ref_df[feature].tolist(),
            'current': curr_df[feature].tolist()
        }
    return jsonify({
        'drift_results': drift_results,
        'drift_data': drift_data
    })

# 예측 결과 모니터링 시각화 API
@dashboard.route('/api/prediction_monitor')
@login_required
@admin_required
def prediction_monitor():
    # 최근 30일간의 예측 결과 데이터
    cutoff_date = datetime.now() - timedelta(days=30)
    results = IrisResult.query.filter(IrisResult.created_at >= cutoff_date).all()
    if not results:
        return jsonify({'data': {'labels': [], 'datasets': []}})

    df = pd.DataFrame([
        {'date': r.created_at.date(), 'predicted_class': r.predicted_class}
        for r in results
    ])
    # 날짜별 예측 클래스 분포 집계
    df['count'] = 1
    pivot_df = pd.pivot_table(df, values='count', index='date', columns='predicted_class', aggfunc='sum', fill_value=0)
    
    #dates = pivot_df.index.strftime('%Y-%m-%d').tolist()
    dates = pd.to_datetime(pivot_df.index).strftime('%Y-%m-%d').tolist() 
    datasets = []
    target_names = ['setosa', 'versicolor', 'virginica']
    
    for cls in target_names:
        if cls in pivot_df.columns:
            datasets.append({
                'label': cls,
                'data': pivot_df[cls].tolist(),
                'fill': False,
                'borderColor': get_color_by_class(cls)
            })

    return jsonify({
        'data': {
            'labels': dates,
            'datasets': datasets
        }
    })

# 예측 이상 감지 및 알림 API
@dashboard.route('/api/anomaly_detection')
@login_required
@admin_required
def anomaly_detection():
    # 최근 100건의 예측 결과
    results = IrisResult.query.order_by(IrisResult.created_at.desc()).limit(100).all()
    if not results or len(results) < 20: # 최소 20건 이상 데이터 필요
        return jsonify({'anomalies': [], 'message': '데이터가 부족합니다.'})
    df = pd.DataFrame([
        {'id': r.id, 'sepal_length': r.sepal_length, 'sepal_width': r.sepal_width, 
         'petal_length': r.petal_length, 'petal_width': r.petal_width, 'predicted_class': r.predicted_class}
        for r in results
    ])
    # IQR (Interquartile Range) 방법을 사용한 간단한 이상치 탐지
    q1 = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].quantile(0.25)
    q3 = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    anomalies = []
    for index, row in df.iterrows():
        is_anomaly = False
        anomaly_features = []
        for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
            if not (lower_bound[feature] <= row[feature] <= upper_bound[feature]):
                is_anomaly = True
                anomaly_features.append(feature)
        
        if is_anomaly:
            anomalies.append({
                'id': row['id'],
                'features': ', '.join(anomaly_features),
                'predicted_class': row['predicted_class']
            })

    return jsonify({
        'anomalies': anomalies,
        'message': f'총 {len(anomalies)}건의 잠재적 이상 데이터가 감지되었습니다.'
    })

# 모델 재학습 및 배포/롤백 API
@dashboard.route('/api/trigger_retrain', methods=['POST'])
@login_required
@admin_required
def trigger_retrain():
    # 실제 환경에서는 백그라운드 작업 큐(Celery 등)로 전송
    from iris.iris_model import train_and_save_model
    try:
        train_and_save_model()
        flash('모델 재학습 트리거가 성공적으로 실행되었습니다. (백그라운드 작업)', 'success')
    except Exception as e:
        flash(f'모델 재학습 실패: {e}', 'danger')

    # 로그 기록
    log = UsageLog(
        user_id=current_user.id,
        service_id=Service.query.filter_by(servicename='Iris').first().id,
        endpoint=request.endpoint,
        usage_type=UsageType.WEB_UI,
        log_status=LogStatusType.CONFIRMED, # 또는 '모델 재학습' 상태 추가
        remote_addr=request.remote_addr,
        request_data_summary='모델 재학습 트리거'
    )
    db.session.add(log)
    db.session.commit()

    return redirect(url_for('dashboard.index'))

@dashboard.route('/api/deploy_model', methods=['POST'])
@login_required
@admin_required
def deploy_model():
    # 실제 CI/CD 파이프라인 연동 로직
    # 예: requests.post('https://jenkins.example.com/job/deploy_model/build', ...)
    flash('모델 배포 트리거가 실행되었습니다. (더미)', 'success')
    return redirect(url_for('dashboard.index'))

@dashboard.route('/api/rollback_model', methods=['POST'])
@login_required
@admin_required
def rollback_model():
    # 실제 CI/CD 파이프라인 연동 로직
    flash('모델 롤백 트리거가 실행되었습니다. (더미)', 'success')
    return redirect(url_for('dashboard.index'))

# Helper function
def get_color_by_class(class_name):
    colors = {
        'setosa': 'rgb(255, 99, 132)',
        'versicolor': 'rgb(54, 162, 235)',
        'virginica': 'rgb(75, 192, 192)',
    }
    return colors.get(class_name, 'rgb(128, 128, 128)')