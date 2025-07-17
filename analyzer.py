import cv2
from pose.detector import get_pose_landmarks, is_arms_raised, is_arms_crossed, is_hands_in_pocket, is_hands_together
from hands.detector import is_v_sign, is_thumbs_up, is_finger_heart
from hands.detector import get_hand_landmarks

def analyze_all_conditions(image_rgb):
    results = []

    # 1. Pose 판단 먼저
    pose_landmarks = get_pose_landmarks(image_rgb)
    if pose_landmarks:
        if is_arms_raised(pose_landmarks):
            results.append("Arms_Raised")
        if is_arms_crossed(pose_landmarks):
            results.append("Arms_Crossed")
        if is_hands_together(pose_landmarks):
            results.append("Hands_Together")
        if is_hands_in_pocket(pose_landmarks):
            results.append("Hands_In_Pocket")

    # 2. Hands 판단 나중
    hand_landmarks_list = get_hand_landmarks(image_rgb)
    if hand_landmarks_list:
        if is_v_sign(hand_landmarks_list):
            results.append("V_Sign")
        if is_thumbs_up(hand_landmarks_list):
            results.append("Thumbs_Up")
        if is_finger_heart(hand_landmarks_list):
            results.append("Finger_Heart")

    return results
