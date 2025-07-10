import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

def detect_v_sign(hand_img):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

        # 수정: multi_hand_landmarks로 변경
        if not results.multi_hand_landmarks:
            return False, {"reason": "No hand detected"}

        landmarks = results.multi_hand_landmarks[0].landmark

        def is_finger_extended(finger_tip, finger_pip):
            return landmarks[finger_tip].y < landmarks[finger_pip].y

        fingers = {
            "index": is_finger_extended(8, 6),
            "middle": is_finger_extended(12, 10),
            "ring": is_finger_extended(16, 14),
            "pinky": is_finger_extended(20, 18),
        }

        is_v = fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
        return is_v, {"fingers": fingers, "is_v": is_v}
