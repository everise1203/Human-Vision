import joblib
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. 데이터 불러오기
print("📂 데이터 불러오는 중...")
X = joblib.load("X_processed.pkl")
y = joblib.load("y_data.pkl")
le = joblib.load("label_encoder.pkl")

# 2. 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ 학습 데이터: {X_train.shape}, 검증 데이터: {X_test.shape}")

# 3. 모델 정의 (진행률 표시 + 멀티코어 사용)
clf = RandomForestClassifier(
    n_estimators=200,
    verbose=1,       # 진행률 출력
    n_jobs=-1,       # 모든 CPU 코어 사용
    random_state=42
)

# 4. 학습
print("🚀 모델 학습 시작...")
start = time.time()
clf.fit(X_train, y_train)
elapsed = time.time() - start
print(f"⏱️ 학습 완료! 소요 시간: {elapsed:.2f}초")

# 5. 평가
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 정확도: {acc * 100:.2f}%")
print("\n📋 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. 모델 저장
joblib.dump(clf, "gesture_classifier.pkl")
print("✅ 모델 저장 완료: gesture_classifier.pkl")