import cv2
import mediapipe as mp
import numpy as np
from finger_state_detector import get_finger_states 
from finger_state_detector import get_gesture_name

# 초기화
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
last_gesture = None
gesture_start_time = None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # 21개 좌표를 (x, y, z) 형태로 리스트에 저장
            h, w, _ = img.shape
            landmarks = []
            for lm in handLms.landmark:
                cx, cy, cz = lm.x * w, lm.y * h, lm.z
                landmarks.append((cx, cy, cz))

            # 손가락 펴짐 상태 확인
            finger_states = get_finger_states(landmarks)
            gesture_name = get_gesture_name(finger_states)
            if gesture_name:
                 print(f"손가락 상태: {finger_states} → {gesture_name}")
            else:
                 continue

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
