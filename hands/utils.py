import math

def is_finger_extended(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_v_sign(landmarks):
    fingers = {
        "index": is_finger_extended(landmarks, 8, 6),
        "middle": is_finger_extended(landmarks, 12, 10),
        "ring": is_finger_extended(landmarks, 16, 14),
        "pinky": is_finger_extended(landmarks, 20, 18),
    }
    is_v = fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
    return is_v, fingers

def is_thumbs_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    other_folded = all(
        landmarks[tip].y > landmarks[pip].y
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )
    thumb_up = thumb_tip.y < thumb_ip.y
    return thumb_up and other_folded

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_finger_heart(landmarks1, landmarks2):
    thumb_dist = euclidean(landmarks1[4], landmarks2[4])
    index_dist = euclidean(landmarks1[8], landmarks2[8])
    return thumb_dist < 0.08 and index_dist < 0.08
