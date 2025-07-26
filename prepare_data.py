import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# 1. CSV 불러오기
df = pd.read_csv("gesture_data.csv")

# 2. 라벨 인코딩
le = LabelEncoder()
y = le.fit_transform(df["label"])  # 문자열 라벨 → 숫자

# 3. Feature 추출
X = df.drop(columns=["label"]).values

# 4. 결과 확인
print(f"샘플 수: {X.shape[0]}")
print(f"특징 차원: {X.shape[1]}")
print(f"라벨 수: {len(set(y))}")
print(f"라벨 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 저장
joblib.dump(X, "X_data.pkl")
joblib.dump(y, "y_data.pkl")
joblib.dump(le, "label_encoder.pkl")
