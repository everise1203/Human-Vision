# 🖐️ Hand Gesture Recognition System (Real-time with MediaPipe + ML)

본 프로젝트는 **노트북 카메라**를 활용하여 **손 제스처를 실시간으로 인식**하는 시스템입니다.  
MediaPipe를 이용해 손의 3D 관절 좌표를 추출하고, Random Forest 기반 머신러닝 모델로 제스처를 분류합니다.

## 📁 프로젝트 구조

```
├── Collect_all_gestures.py        # 손 제스처 데이터 수집
├── prepare_data.py                # CSV → numpy + 라벨 인코딩
├── preprocess_utils.py            # 전처리 및 증강 함수
├── Data_preprocessing.py          # 데이터 전처리 및 저장
├── handgesture_train_model.py     # RandomForestClassifier 학습
├── handgesture_realtime_inference.py  # 실시간 추론
├── hand_tracking_test.py          # 손가락 펴짐 상태 기반 단순 분류
├── finger_state_detector.py       # 손가락 굽힘/펴짐 판단 및 제스처 매핑
├── gesture_data.csv               # 수집된 원시 데이터
├── X_data.pkl, y_data.pkl         # 전처리 전 데이터
├── X_processed.pkl                # 전처리 완료 데이터
├── gesture_classifier.pkl         # 학습된 모델
├── label_encoder.pkl              # 라벨 인코더
```

---

## 🛠️ 실행 방법

### 1. 손 제스처 데이터 수집
```bash
python Collect_all_gestures.py
```
- 손을 카메라 앞에 보여주고, 지시에 따라 데이터를 수집합니다.
- 종료: `q` 키

### 2. 데이터 준비 및 전처리
```bash
python prepare_data.py
python Data_preprocessing.py
```

### 3. 모델 학습
```bash
python handgesture_train_model.py
```
- 학습 완료 후 `gesture_classifier.pkl` 저장됨.

### 4. 실시간 추론 실행
```bash
python handgesture_realtime_inference.py
```
- 웹캠에서 손 제스처를 인식하고 화면에 결과를 표시합니다.

### (옵션) 단순 손가락 상태 기반 테스트
```bash
python hand_tracking_test.py
```

---

## 🧠 사용 기술

| 항목 | 내용 |
|------|------|
| 손 추적 | [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) |
| 영상 처리 | OpenCV |
| 모델 | RandomForestClassifier (sklearn) |
| 전처리 | 상대좌표 + 정규화 + jitter noise |
| 추론 안정화 | `deque`를 활용한 예측 누적 투표 |

---

## 📊 제스처 라벨 (예시)
- `A Shape`: [1,0,0,0,0]
- `B Shape`: [0,1,0,0,0]
- `J Shape`: [0,1,1,1,0]
- 등 총 15개 제스처 지원
---

## 📄 라이선스
MIT License
