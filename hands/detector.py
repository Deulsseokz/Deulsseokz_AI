import mediapipe as mp
import cv2

from hands.utils import is_v_sign, is_thumbs_up, is_finger_heart

mp_hands = mp.solutions.hands

def detect_pose(hand_img):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        results = hands.process(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return {"pose": "None", "reason": "No hand detected"}

        hands_landmarks = results.multi_hand_landmarks
        poses = []

        # 하트 포즈는 두 손 필요
        if len(hands_landmarks) == 2:
            if is_finger_heart(hands_landmarks[0].landmark, hands_landmarks[1].landmark):
                return {"pose": "Heart"}

        for hand in hands_landmarks:
            lm = hand.landmark
            is_v, fingers = is_v_sign(lm)
            if is_v:
                poses.append({"pose": "V sign", "fingers": fingers})
                continue
            if is_thumbs_up(lm):
                poses.append({"pose": "Thumbs up"})
                continue
            poses.append({"pose": "Unknown"})

        return {"results": poses}