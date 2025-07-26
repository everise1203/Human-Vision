import cv2
import numpy as np
import mediapipe as mp
import joblib
from preprocess_utils import augment_landmarks, preprocess_landmarks  
from collections import deque

# 모델과 인코더 불러오기
clf = joblib.load("gesture_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Mediapipe 셋업
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 추론 안정성을 위한 이력 큐
pred_history = deque(maxlen=10)

# 카메라 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 미러 효과 + RGB 변환
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Landmark 추출
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)

            # 전처리
            lm_aug = augment_landmarks(landmarks)
            lm_proc = preprocess_landmarks(lm_aug).reshape(1, -1)

            # 예측
            pred = clf.predict(lm_proc)[0]
            label = le.inverse_transform([pred])[0]
            pred_history.append(label)

            # 가장 많이 등장한 예측 결과 출력
            if len(pred_history) == pred_history.maxlen:
                final_pred = max(set(pred_history), key=pred_history.count)
                cv2.putText(frame, f'Gesture: {final_pred}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # 랜드마크 시각화
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
