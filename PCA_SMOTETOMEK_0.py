import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# 평가 함수
def eval_score(y_valid, y_pred_class):
    # 오차행렬 출력
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, y_pred_class))
    
    # 분류 보고서 출력 (Recall 값에 집중)
    recall = recall_score(y_valid, y_pred_class)
    print(f"\nRecall: {recall:.4f}")
    
    # 추가 평가 결과를 원할 경우 classification_report 사용
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred_class))

# 모델 학습 및 평가 함수 (특정 모델만 실행)
def run_model_on_data(df, file_name, model_name):
    print(f"\n=== Processing data from {file_name} with {model_name} ===")
    
    # Feature와 Target 분리
    X = df.drop('target', axis=1)
    y = df['target']

    # 데이터 스케일링 (PCA 적용 전)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 적용 (분산을 95% 유지)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # 훈련/테스트 데이터 분할 (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE-Tomek 적용
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

    # 모델 리스트 (필요한 모델만 선택 가능)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # 모델 실행
    if model_name in models:
        model = models[model_name]
        model.fit(X_train_resampled, y_train_resampled)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        eval_score(y_test, y_pred_class)
    else:
        print(f"Model {model_name} is not available. Please choose a valid model.")

# 파일 1: 0_drop_regression.csv 불러오기
df_0_drop_regression = pd.read_csv('0_drop_regression.csv')
# 실행 예시: Logistic Regression 사용
run_model_on_data(df_0_drop_regression, "0_drop_regression.csv", "Logistic Regression")
# 실행 예시: XGBoost 사용
run_model_on_data(df_0_drop_regression, "0_drop_regression.csv", "XGBoost")
# 실행 예시: LightGBM 사용
run_model_on_data(df_0_drop_regression, "0_drop_regression.csv", "LightGBM")
# 실행 예시: CatBoost 사용
run_model_on_data(df_0_drop_regression, "0_drop_regression.csv", "CatBoost")
# 실행 예시: Random Forest 사용
run_model_on_data(df_0_drop_regression, "0_drop_regression.csv", "Random Forest")

# 파일 2: 0_regression.csv 불러오기
df_0_regression = pd.read_csv('0_regression.csv')
# 실행 예시: Logistic Regression 사용
run_model_on_data(df_0_regression, "0_regression.csv", "Logistic Regression")
# 실행 예시: XGBoost 사용
run_model_on_data(df_0_regression, "0_regression.csv", "XGBoost")
# 실행 예시: LightGBM 사용
run_model_on_data(df_0_regression, "0_regression.csv", "LightGBM")
# 실행 예시: CatBoost 사용
run_model_on_data(df_0_regression, "0_regression.csv", "CatBoost")
# 실행 예시: Random Forest 사용
run_model_on_data(df_0_regression, "0_regression.csv", "Random Forest")

# 파일 3: 0_drop.csv 불러오기
df_0_drop = pd.read_csv('0_drop.csv')
# 실행 예시: Logistic Regression 사용
run_model_on_data(df_0_drop, "0_drop.csv", "Logistic Regression")
# 실행 예시: XGBoost 사용
run_model_on_data(df_0_drop, "0_drop.csv", "XGBoost")
# 실행 예시: LightGBM 사용
run_model_on_data(df_0_drop, "0_drop.csv", "LightGBM")
# 실행 예시: CatBoost 사용
run_model_on_data(df_0_drop, "0_drop.csv", "CatBoost")
# 실행 예시: Random Forest 사용
run_model_on_data(df_0_drop, "0_drop.csv", "Random Forest")
