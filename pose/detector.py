import mediapipe as mp
import math

mp_pose = mp.solutions.pose

def get_pose_landmarks(image_rgb):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark

def is_arms_raised(landmarks):
    return landmarks[15].y < landmarks[11].y and landmarks[16].y < landmarks[12].y

def is_hands_in_pocket(landmarks):
    lw, rw, lh, rh = landmarks[15], landmarks[16], landmarks[23], landmarks[24]
    return lw.y > lh.y and abs(lw.x - lh.x) < 0.1 and rw.y > rh.y and abs(rw.x - rh.x) < 0.1

def is_arms_crossed(landmarks):
    le, re, lw, rw = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    ed = ((le.x - re.x)**2 + (le.y - re.y)**2)**0.5
    wd = ((lw.x - rw.x)**2 + (lw.y - rw.y)**2)**0.5
    return ed < 0.15 and wd < 0.15

def is_hands_together(landmarks):
    lw, rw = landmarks[15], landmarks[16]
    return ((lw.x - rw.x)**2 + (lw.y - rw.y)**2)**0.5 < 0.07