import pandas as pd

df_oil = pd.read_csv("data_week3.csv")
df = df_oil.copy()

# 1. 각 변수의 고유 값 확인
print("각 변수의 고유 값 개수 확인:")
for col in df.columns:
    unique_values = df[col].unique()
    print(f"{col}의 고유 값: {len(unique_values)}개")

# 2. 고유 값이 적은 변수를 범주형으로 변환하는 코드
# 특정 기준으로 고유 값이 적은 경우 (여기서는 10개 이하인 경우), 범주형으로 변환
for col in df.columns:
    if len(df[col].unique()) <= 10:  # 고유 값이 10개 이하인 경우에만 범주형 변환
        df[col] = df[col].astype('category')
        print(f"{col} 변수를 범주형으로 변환했습니다.")

# 3. 변환된 후 변수들의 데이터 타입 확인
print("\n변환 후 각 변수의 데이터 타입 확인:")
print(df.dtypes)

# 4. 범주형 변수를 더미 변수로 변환 (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first=True로 차원을 줄임
print("\nOne-Hot Encoding 후 데이터:")
print(df_encoded.head())

df.info()