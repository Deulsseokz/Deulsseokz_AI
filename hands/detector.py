import mediapipe as mp
import numpy as np
import cv2

# Hand Landmarker 모델 초기화
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)

DEFINED_HAND_KEYWORDS = ["PeaceSign", "HeartPose", "Point", "HandsTogether"]

def analyze_hand_gesture(image_bgr: np.ndarray, keyword: str) -> bool:
    if keyword not in DEFINED_HAND_KEYWORDS:
        print(f"✅ Unknown hand gesture keyword: '{keyword}'. Returning True as requested.")
        return True

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands_detector.process(image_rgb)
    image_rgb.flags.writeable = True

    if not results.multi_hand_landmarks:
        print("⚠️ [analyze_hands] 이미지에서 손을 찾을 수 없습니다.")
        return False

    # 모든 감지된 손에 대해 규칙 확인
    for hand_landmarks in results.multi_hand_landmarks:
        if keyword == "PeaceSign":
            # 검지와 중지는 펴고, 약지와 새끼손가락은 구부렸는지 확인
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

            if (index_finger_tip.y < index_finger_pip.y and
                    middle_finger_tip.y < middle_finger_pip.y and
                    ring_finger_tip.y > ring_finger_pip.y):
                return True

        elif keyword == "HeartPose":
            # 엄지 끝과 검지 끝이 가까이 있는지 확인 (손가락 하트)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
            if distance < 0.1:  # 임계값 조정 필요
                return True

        elif keyword == "Point":
            # 검지는 펴고, 나머지 손가락은 구부렸는지 확인
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            if (index_finger_tip.y < index_finger_pip.y and
                    middle_finger_tip.y > middle_finger_pip.y):
                return True

    # HandsTogether는 양손이 필요하므로 루프 밖에서 처리
    if keyword == "HandsTogether":
        if len(results.multi_hand_landmarks) == 2:
            wrist1 = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
            wrist2 = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST]
            distance = np.linalg.norm(np.array([wrist1.x, wrist1.y]) - np.array([wrist2.x, wrist2.y]))
            if distance < 0.2:
                return True

    return False