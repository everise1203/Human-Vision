import cv2
import mediapipe as mp
import csv
import os

# 제스처 라벨 목록 (A ~ O)
GESTURE_LABELS = [
    "J Shape"
]

SAVE_PATH = "gesture_data.csv"
MAX_FRAMES_PER_GESTURE = 1000

# MediaPipe 손 추적 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# CSV 파일이 없으면 헤더 작성
if not os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

# 카메라 시작
cap = cv2.VideoCapture(0)

# 제스처별 수집 반복
for label in GESTURE_LABELS:
    print(f"\n===== {label} 수집을 준비해주세요 =====")
    input("🖐 손을 준비하고 Enter를 누르면 수집이 시작됩니다...")

    collected_frames = []
    frame_count = 0

    while frame_count < MAX_FRAMES_PER_GESTURE:
        success, img = cap.read()
        if not success:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                h, w, _ = img.shape
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                collected_frames.append(landmarks)
                frame_count += 1

                print(f"[{label}] {frame_count}/{MAX_FRAMES_PER_GESTURE} 프레임 수집 중...", end="\r")

        # 상태 표시
        status_text = f"[{label}] {frame_count}/{MAX_FRAMES_PER_GESTURE} collecting..."
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.imshow("Gesture Collector", img)

        # 강제 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n❌ 수집 강제 종료됨")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # 수집 완료 → CSV 저장
    with open(SAVE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        for frame in collected_frames:
            writer.writerow(frame + [label])

    print(f"\n✅ {label} 수집 완료! 총 {frame_count} 프레임 저장됨.")

    # OpenCV 창 닫고 다음 제스처 대기
    cv2.destroyAllWindows()
    input("👉 다음 제스처로 넘어가려면 Enter를 누르세요.")

# 전체 완료
print("\n🎉 모든 제스처(A~O) 수집이 완료되었습니다!")
cap.release()
cv2.destroyAllWindows()
