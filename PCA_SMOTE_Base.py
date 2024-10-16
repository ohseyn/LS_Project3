import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# 데이터 로드
df_oil = pd.read_csv("data_week3.csv")

df = df_oil.copy()

df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6','x7', 'x8', 'x9', 'x10', 'x11',
              'x12', 'x13', 'x14', 'x15', 'x16','x17', 'target']

# 범주형 변수 'x1'을 One-Hot Encoding으로 변환
df = pd.get_dummies(df, columns=['x1'], drop_first=True)

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

# 로지스틱 회귀 모델 생성 및 평가
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_smote, y_train_smote)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)

print("\n=== Logistic Regression ===")
eval_score(y_test, y_pred_class)

# XGBoost 모델 생성 및 평가
model = XGBClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)

print("\n=== XGBoost ===")
eval_score(y_test, y_pred_class)

# LightGBM 모델 생성 및 평가
model = LGBMClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)

print("\n=== LightGBM ===")
eval_score(y_test, y_pred_class)

# CatBoost 모델 생성 및 평가
model = CatBoostClassifier(random_state=42, verbose=0)  # verbose=0으로 학습 과정 출력 방지
model.fit(X_train_smote, y_train_smote)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)

print("\n=== CatBoost ===")
eval_score(y_test, y_pred_class)

# 랜덤 포레스트 모델 생성 및 평가
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)

print("\n=== Random Forest ===")
eval_score(y_test, y_pred_class)