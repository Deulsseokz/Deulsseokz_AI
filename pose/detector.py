# pose/detector.py 에 추가
import numpy as np


def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산합니다. b가 중심점입니다."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# is_arms_raised 함수를 각도 기반으로 개선
def is_arms_raised_upgraded(landmarks):
    # 오른쪽 팔꿈치와 왼쪽 팔꿈치가 어깨보다 위에 있는지 확인
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

    # 오른쪽 팔이 얼마나 펴져 있는지 (어깨-팔꿈치-손목)
    right_arm_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    )

    # 왼쪽 팔이 얼마나 펴져 있는지
    left_arm_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    )

    # 팔꿈치가 어깨보다 위에 있고, 팔이 150도 이상 펴져 있다면 만세로 인정
    if right_elbow.y < right_shoulder.y and left_elbow.y < left_shoulder.y and right_arm_angle > 150 and left_arm_angle > 150:
        return True
    return False

