from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df_raw = pd.read_csv('data_week3.csv')
df = df_raw.copy()

# 컬럼 이름 변경
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
              'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'target']

# 'x4' 문자열로 변환
df['x4'] = df['x4'].astype(str)

# 'x5' 범주화 및 float 변환
cut_lim = [-1, 0, 28]
labels = [0, 1]
df["x5"] = pd.cut(df["x5"], bins=cut_lim, labels=labels).astype(float)

# 범주형 변수 원-핫 인코딩
df = pd.get_dummies(df, columns=['x1', 'x4', 'x5'], drop_first=True)

# 데이터와 타겟 변수 분리
X = df.drop('target', axis=1)
y = df['target']

# 8:2로 데이터 분할 (stratify=y로 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 샘플링 전후 클래스 분포 출력
print("학습 데이터 클래스 분포:")
print(y_train.value_counts())

# 평가 함수 정의
def eval_score(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"정확도: {accuracy:.4f}")
    print(f"재현률: {recall:.4f}")
    print(classification_report(y_true, y_pred))

# Logistic Regression + 비용 민감 학습 적용
lr_model = LogisticRegression(random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)

# 임계값 0.4로 조정
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
y_pred_class = (y_pred_proba >= 0.4).astype(int)  # 기본 0.5에서 0.4로 변경

print("\nLogistic Regression 결과 (임계값 조정 적용):")
eval_score(y_test, y_pred_class)