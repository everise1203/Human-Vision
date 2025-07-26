import numpy as np

# 마디 좌표 21개를 받아서 손가락이 펴졌는지 여부를 판단
def get_finger_states(landmarks):
    """
    landmarks: MediaPipe가 반환하는 21개의 손 마디 좌표 리스트 (x, y, z)

    return: [1, 1, 0, 0, 0] 형식의 리스트 (1 = 펴짐, 0 = 굽힘)
    """
    finger_states = []

    # 엄지: x축 방향으로 비교 (다른 손가락은 y축 기준)
    thumb_is_open = landmarks[4][0] > landmarks[3][0]
    finger_states.append(int(thumb_is_open))

    # 검지~새끼: tip과 pip의 y 좌표 비교
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):
        is_open = landmarks[tip][1] < landmarks[pip][1]
        finger_states.append(int(is_open))

    return finger_states

def get_gesture_name(finger_states):
    """
    finger_states: [1, 0, 0, 0, 0] 형식의 손가락 상태 리스트

    return: 대응되는 제스처 이름 (예: 'A Shape')
    """
    gesture_map = {
        (1,0,0,0,0): "A Shape",
        (0,1,0,0,0): "B Shape",
        (0,0,0,0,1): "C Shape",
        (1,1,0,0,0): "D Shape",
        (1,0,0,0,1): "E Shape",
        (0,1,1,0,0): "F Shape",
        (0,1,0,0,1): "G Shape",
        (1,1,1,0,0): "H Shape",
        (1,1,0,0,1): "I Shape",
        (0,1,1,1,0): "J Shape",
        (0,1,1,0,1): "K Shape",
        (0,1,0,1,1): "L Shape",
        (0,0,1,1,1): "M Shape",
        (0,1,1,1,1): "N Shape",
        (1,1,1,1,1): "O Shape",
    }

    return gesture_map.get(tuple(finger_states), None)
