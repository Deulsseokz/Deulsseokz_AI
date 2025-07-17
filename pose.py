import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def detect_pose_and_crop_hands(image):
    hands_images = []
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return []

        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark
        for wrist_id in [15, 16]:  # LEFT_WRIST, RIGHT_WRIST
            x = int(landmarks[wrist_id].x * w)
            y = int(landmarks[wrist_id].y * h)
            size = 80  # crop size
            hand_crop = image[max(0,y-size):y+size, max(0,x-size):x+size]
            if hand_crop.size > 0:
                hands_images.append(hand_crop)

    return hands_images