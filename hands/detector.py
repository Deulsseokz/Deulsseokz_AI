import mediapipe as mp
import cv2
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions.hands import HandLandmark

mp_hands = mp.solutions.hands

def get_hand_landmarks(image_rgb):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
    return None

def is_v_sign(hand_landmarks_list):
    """
    Mediapipe Hands 결과 (landmark 리스트)로부터 V 사인을 판단
    - 한 손 또는 양손 모두 가능
    """

    def check_single_hand(hand: NormalizedLandmarkList) -> bool:
        index_tip = hand.landmark[HandLandmark.INDEX_FINGER_TIP]
        index_dip = hand.landmark[HandLandmark.INDEX_FINGER_DIP]
        middle_tip = hand.landmark[HandLandmark.MIDDLE_FINGER_TIP]
        middle_dip = hand.landmark[HandLandmark.MIDDLE_FINGER_DIP]
        ring_tip = hand.landmark[HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand.landmark[HandLandmark.PINKY_TIP]

        index_straight = index_tip.y < index_dip.y
        middle_straight = middle_tip.y < middle_dip.y
        ring_folded = ring_tip.y > middle_dip.y
        pinky_folded = pinky_tip.y > middle_dip.y

        return index_straight and middle_straight and ring_folded and pinky_folded

    # 하나라도 만족하면 V 사인으로 판단
    for hand_landmarks in hand_landmarks_list:
        if check_single_hand(hand_landmarks):
            return True

    return False

def is_thumbs_up(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return []
        poses = []
        for lm in results.multi_hand_landmarks:
            thumb = lm.landmark[4].y < lm.landmark[3].y
            others = all(lm.landmark[tip].y > lm.landmark[pip].y for tip, pip in [(8,6), (12,10), (16,14), (20,18)])
            if thumb and others:
                poses.append("Thumbs Up")
        return poses

def is_finger_heart(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            return []
        l1, l2 = results.multi_hand_landmarks[0].landmark, results.multi_hand_landmarks[1].landmark
        dist_thumb = ((l1[4].x - l2[4].x)**2 + (l1[4].y - l2[4].y)**2)**0.5
        dist_index = ((l1[8].x - l2[8].x)**2 + (l1[8].y - l2[8].y)**2)**0.5
        if dist_thumb < 0.08 and dist_index < 0.08:
            return ["Finger Heart"]
        return []
