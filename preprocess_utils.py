import numpy as np
import joblib
from tqdm import tqdm

# 손끝 인덱스
TIP_IDXS = [4, 8, 12, 16, 20]

def augment_landmarks(landmarks, jitter_std=0.01):
    """
    손끝 landmark에 노이즈 추가 (jitter)
    """
    landmarks = landmarks.reshape(-1, 3)
    for idx in TIP_IDXS:
        noise = np.random.normal(0, jitter_std, 3)
        landmarks[idx] += noise
    return landmarks

def preprocess_landmarks(landmarks):
    """
    기준점 기준 상대좌표화 + 정규화
    """
    base = landmarks[0]
    relative = landmarks - base
    max_norm = np.max(np.linalg.norm(relative, axis=1))
    if max_norm > 0:
        normalized = relative / max_norm
    else:
        normalized = relative
    return normalized.flatten()

import joblib
from tqdm import tqdm

# 1. 데이터 불러오기
X = joblib.load("X_data.pkl")
y = joblib.load("y_data.pkl")

# 2. 전처리 적용
X_processed = []

for row in tqdm(X, desc="전처리 중"):
    lm = np.array(row).reshape(-1, 3)
    lm_aug = augment_landmarks(lm)  # 증강
    lm_proc = preprocess_landmarks(lm_aug)  # 전처리
    X_processed.append(lm_proc)

X_processed = np.array(X_processed)

# 3. 저장
joblib.dump(X_processed, "X_processed.pkl")
print("✅ 전처리 완료 및 저장됨: X_processed.pkl")
