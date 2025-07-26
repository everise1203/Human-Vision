import numpy as np

TIP_IDXS = [4, 8, 12, 16, 20]  # 손끝 인덱스

def augment_landmarks(landmarks, jitter_std=0.01):
    """
    손끝 landmark에 노이즈 추가 (jitter)
    """
    landmarks = np.array(landmarks).reshape(-1, 3)
    for idx in TIP_IDXS:
        noise = np.random.normal(0, jitter_std, 3)
        landmarks[idx] += noise
    return landmarks

def preprocess_landmarks(landmarks):
    """
    상대좌표화 + 정규화
    """
    base = landmarks[0]  # wrist 기준
    relative = landmarks - base

    max_norm = np.max(np.linalg.norm(relative, axis=1))
    if max_norm > 0:
        normalized = relative / max_norm
    else:
        normalized = relative

    return normalized.flatten()
