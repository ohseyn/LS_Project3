import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# 평가 함수
def eval_score(y_valid, y_pred_class):
    confusion = confusion_matrix(y_valid, y_pred_class)
    accuracy = accuracy_score(y_valid, y_pred_class)
    precision = precision_score(y_valid, y_pred_class)
    recall = recall_score(y_valid, y_pred_class)
    f1 = f1_score(y_valid, y_pred_class)
    print("오차행렬")
    print(confusion)
    print("정확도: {:.4f}, 정밀도: {:.4f}, 재현율: {:.4f}, F1: {:.4f}".format(accuracy, precision, recall, f1))

# 모델 학습 및 평가 함수
def run_models_on_data(df, file_name):
    print(f"\n=== Processing data from {file_name} ===")
    
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

    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 모델 리스트 (로지스틱 회귀, XGBoost, LightGBM, CatBoost, 랜덤 포레스트)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # 각 모델별로 학습 및 평가
    for model_name, model in models.items():
        model.fit(X_train_smote, y_train_smote)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        print(f"\n=== {model_name} ===")
        eval_score(y_test, y_pred_class)

# 파일 1: 0_drop_regression.csv
df_0_drop_regression = pd.read_csv('0_drop_regression.csv')
run_models_on_data(df_0_drop_regression, "0_drop_regression.csv")

# 파일 2: 0_regression.csv
df_0_regression = pd.read_csv('0_regression.csv')
run_models_on_data(df_0_regression, "0_regression.csv")

# 파일 3: 0_drop.csv
df_0_drop = pd.read_csv('0_drop.csv')
run_models_on_data(df_0_drop, "0_drop.csv")
