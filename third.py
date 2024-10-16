import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns

df_oil = pd.read_csv("data_week3.csv")

df = df_oil.copy()

df

df.info()

df.describe()

df.isna().sum()

df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6','x7', 'x8', 'x9', 'x10', 'x11',
    'x12', 'x13', 'x14', 'x15', 'x16','x17', 'target']

# 교차표 생성
contingency_table = pd.crosstab(df['x1'], df['target'])
print("교차표 (실제 빈도):")
print(contingency_table)

# 카이제곱 검정 실행
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 결과 출력
print("\n카이제곱 통계량: ", chi2)
print("p-값: ", p)
print("자유도: ", dof)
print("\n기대 빈도 (expected 빈도):")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

# 실제 빈도와 기대 빈도를 시각적으로 비교하기 위해 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 실제 빈도 그래프
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues")
plt.title("실제 빈도 Heatmap")
plt.show()

# 기대 빈도 그래프
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns),
            annot=True, fmt=".2f", cmap="Greens")
plt.title("기대 빈도 Heatmap")
plt.show()

#================================================
# unknown1의 type별로 카이제곱 검정 결과 확인
chi2_results_by_type = {}

# 각 type별로 나눠서 카이제곱 검정을 변수와 각각 진행
types = df['x1'].unique()  # unknown1의 고유 값들
for t in types:
    chi2_results_by_type[t] = {}
    # 해당 type만 선택한 데이터셋
    subset = df[df['x1'] == t]
    
    # unknown2 ~ unknown17까지의 변수와 카이제곱 검정
    for col in df.columns[1:-1]:  # unknown2 ~ unknown17
        try:
            # 교차표 생성
            contingency_table = pd.crosstab(subset[col], subset['target'])
            # 카이제곱 검정 실행
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            chi2_results_by_type[t][col] = {'chi2': chi2, 'p-value': p, 'dof': dof}
        except ValueError:
            # 카이제곱 검정이 불가능한 경우를 처리
            chi2_results_by_type[t][col] = 'Chi-square test not possible'

# 각 type별로 나눠서 카이제곱 검정을 변수와 각각 진행 후 출력
def show_all_chi2_results():
    for t in chi2_results_by_type:
        print(f"\n=== Results for {t} ===")
        for col, result in chi2_results_by_type[t].items():
            print(f"\nVariable: {col}")
            if isinstance(result, dict):
                print(f"  - Chi2: {result['chi2']:.4f}")
                print(f"  - p-value: {result['p-value']:.4e}")  # 지수 표기법으로 p-value를 깔끔하게 출력
                print(f"  - Degrees of Freedom: {result['dof']}")
            else:
                print(f"  {result}")
                
# 결과를 확인하기 위한 실행
show_all_chi2_results()

sns.countplot(x='x1', data=df)

