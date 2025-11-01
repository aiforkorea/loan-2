# apps/iris/iris_model.py
# 모델 준비 (최초 1회 실행) --> 1회 실행을 통한 iris/model.pkl 생성 및 model.pkl 추론 활용 
# 1. pickle 이용
#from sklearn.datasets import load_iris
#from sklearn.linear_model import LogisticRegression
#import pickle
#iris = load_iris()
#model = LogisticRegression(max_iter=200)
#model.fit(iris.data, iris.target)
#with open('apps/iris/model.pkl', 'wb') as f:
#    pickle.dump(model, f)
# 2. joblib + 함수화를 통한 model.pkl 수시 생성
import joblib, os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def train_and_save_model():
    # 1 데이터 로드
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 2 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # 4 훈련 데이터 성능 평가
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    print("--- 훈련 데이터 성능 ---")
    print(f"혼동 행렬:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"정확도: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"정밀도 (가중 평균): {precision_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"재현율 (가중 평균): {recall_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"F1-점수 (가중 평균): {f1_score(y_train, y_train_pred, average='weighted'):.4f}")
    print(f"AUC-ROC (가중 평균): {roc_auc_score(y_train, y_train_prob, multi_class='ovr', average='weighted'):.4f}")
    print("-" * 25)
    # 5 테스트 데이터 성능 평가
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    print("--- 테스트 데이터 성능 ---")
    print(f"혼동 행렬:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"정확도: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"정밀도 (가중 평균): {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"재현율 (가중 평균): {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"F1-점수 (가중 평균): {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"AUC-ROC (가중 평균): {roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted'):.4f}")
    print("-" * 25)
    # 6 모델 저장
    if not os.path.exists('apps/iris'):
        os.makedirs('apps/iris')
    joblib.dump(model, 'apps/iris/model.pkl')
    print("Iris model trained and saved to apps/iris/model.pkl")
if __name__ == '__main__':
    train_and_save_model()
