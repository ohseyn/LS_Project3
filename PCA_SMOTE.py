import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df_oil = pd.read_csv("data_week3.csv")

df = df_oil.copy()

df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6','x7', 'x8', 'x9', 'x10', 'x11',
    'x12', 'x13', 'x14', 'x15', 'x16','x17', 'target']

df

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

# 최종 데이터 출력
print("훈련 데이터 크기:", X_train_smote.shape)
print("테스트 데이터 크기:", X_test.shape)
print("SMOTE 이후 클래스 분포:\n", pd.Series(y_train_smote).value_counts())

# 훈련 데이터와 테스트 데이터를 CSV로 저장
train_data_smote = pd.DataFrame(X_train_smote)
train_data_smote['target'] = y_train_smote
train_data_smote.to_csv('train_data_pca_smote.csv', index=False)

test_data = pd.DataFrame(X_test)
test_data['target'] = y_test.values  # numpy array에서 DataFrame으로 변환
test_data.to_csv('test_data_pca_smote.csv', index=False)