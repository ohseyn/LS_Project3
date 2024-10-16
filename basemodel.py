import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import os

os.getcwd()

raw_data = pd.read_csv("project3/data/data_week3.csv")
df = raw_data.copy()

df.info()
df.describe()
df.head()

df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6','x7', 'x8', 'x9', 'x10', 'x11',
    'x12', 'x13', 'x14', 'x15', 'x16','x17', 'target']

df["target"].hist() # 불균형 데이터

len(df[df["target"] == 0]) # 0 갯수
len(df[df["target"] == 1]) # 1 갯수

# target = 0인 비율
ratio = (len(df[df["target"] == 0])/(len(df[df["target"] == 0]) + len(df[df["target"] == 1]))) * 100

# 정수로 변환
ratio = int(ratio)

print(f'0의 비율: {ratio}%')

# 결측치 확인
df.isna().sum()

# unknown1 값 확인
df["x1"].unique()

# 범주형 데이터 분포 확인
df["x1"].hist()

# x1 원핫 인코딩 하기

df = pd.get_dummies(data = df, columns = ["x1"], drop_first = True)

X = df.drop("target", axis = 1)
y = df["target"]

# 학습용 데이터와 테스트용 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

# 로지스틱 회귀 모델 생성
model = LogisticRegression(max_iter=1000, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 예측 확률값 계산 (양성 클래스에 대한 확률)
y_pred_proba = model.predict_proba(X_valid)[:, 1]

# 0.5이상값 1로 변환
y_pred_class = (y_pred_proba >= 0.5).astype(int)

def eval_score(y_valid, y_pred_class):
    confusion = confusion_matrix(y_valid, y_pred_class)
    accuracy = accuracy_score(y_valid, y_pred_class)
    precision = precision_score(y_valid, y_pred_class)
    recall = recall_score(y_valid, y_pred_class)
    f1 = f1_score(y_valid, y_pred_class)
    print("오차행렬")
    print(confusion)
    print("정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}".format(accuracy = accuracy, precision = precision, recall = recall, f1 = f1))

eval_score(y_valid, y_pred_class)

# XGBoost
model = XGBClassifier(random_state = 42)

model.fit(X_train, y_train)


y_pred_proba = model.predict_proba(X_valid)[:, 1]

y_pred_class = (y_pred_proba >= 0.5).astype(int)

eval_score(y_valid, y_pred_class)


# LGBM

model = LGBMClassifier(random_state = 42)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_valid)[:, 1]

y_pred_class = (y_pred_proba >= 0.5).astype(int)

eval_score(y_valid, y_pred_class)


# CatBoost
model = CatBoostClassifier(random_state = 42)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_valid)[:, 1]

y_pred_class = (y_pred_proba >= 0.5).astype(int)

eval_score(y_valid, y_pred_class)


# Random Forest
model = RandomForestClassifier(random_state = 42)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_valid)[:, 1]

y_pred_class = (y_pred_proba >= 0.5).astype(int)

eval_score(y_valid, y_pred_class)
