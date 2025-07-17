import math
from mediapipe.python.solutions.hands import HandLandmark

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_finger_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_v_sign(hand_landmarks_list):
    def check_single_hand(hand):
        index_tip = hand[HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand[HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand[HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand[HandLandmark.PINKY_TIP]

        index_pip = hand[HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand[HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand[HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand[HandLandmark.PINKY_PIP]

        is_index_up = index_tip.y < index_pip.y
        is_middle_up = middle_tip.y < middle_pip.y
        is_ring_folded = ring_tip.y > ring_pip.y
        is_pinky_folded = pinky_tip.y > pinky_pip.y

        return is_index_up and is_middle_up and is_ring_folded and is_pinky_folded

    for hand_landmarks in hand_landmarks_list:
        if check_single_hand(hand_landmarks):
            return True
    return False

def is_thumbs_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    folded = all(landmarks[tip].y > landmarks[pip].y for tip, pip in [(8,6),(12,10),(16,14),(20,18)])
    return thumb_tip.y < thumb_ip.y and folded

def is_finger_heart(lm1, lm2):
    return euclidean(lm1[4], lm2[4]) < 0.08 and euclidean(lm1[8], lm2[8]) < 0.08