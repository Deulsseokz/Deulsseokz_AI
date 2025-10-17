import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """세 점 a, b, c 사이의 각도를 계산하는 헬퍼 함수"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_pose(image_bgr: np.ndarray, keyword: str) -> bool:
    # with 구문을 사용해 리소스를 안전하게 관리합니다.
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            print("⚠️ [analyze_pose] 이미지에서 사람의 형체를 찾을 수 없습니다.")
            return False

        try:
            if keyword == "StrongPose":
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                is_hand_on_hip_l = hip_l[1] > wrist_l[1] > shoulder_l[1]
                is_hand_on_hip_r = hip_r[1] > wrist_r[1] > shoulder_r[1]
                return (60 < angle_l < 140 and 60 < angle_r < 140) and (is_hand_on_hip_l and is_hand_on_hip_r)

            elif keyword == "Sitting":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)
                return 70 < angle < 140

            elif keyword == "Jump":
                left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                # 양 발목이 엉덩이보다 위쪽에 있으면 점프로 판단
                return left_ankle_y < left_hip_y and right_ankle_y < right_hip_y

            elif keyword in ["Arms Up", "Spreading Arms"]:
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                angle_l = calculate_angle(hip_l, shoulder_l, elbow_l)
                angle_r = calculate_angle(hip_r, shoulder_r, elbow_r)

                if keyword == "Arms Up": # 만세
                    return angle_l > 140 and angle_r > 140
                else: # Spreading Arms
                    return 70 < angle_l < 130 and 70 < angle_r < 130

            elif keyword == "Deep Bow":
                shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
                hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
                return shoulder_y > hip_y

            elif keyword == "Shielding Eyes":
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                dist_l = np.linalg.norm(np.array(wrist_l) - np.array(nose))
                dist_r = np.linalg.norm(np.array(wrist_r) - np.array(nose))
                return dist_l < 0.15 or dist_r < 0.15

            elif keyword == "Back View":
                return landmarks[mp_pose.PoseLandmark.NOSE.value].visibility < 0.5

            elif keyword == "Crossing Arms":
                wrist_l_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
                wrist_r_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
                shoulder_l_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                shoulder_r_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                return wrist_l_x > wrist_r_x and wrist_l_x < shoulder_r_x and wrist_r_x > shoulder_l_x

            elif keyword == "CreativePose":
                return True # 항상 성공

            else:
                print(f"⚠️ Unknown pose keyword: '{keyword}'. Returning True as default.")
                return True

        except Exception as e:
            # 랜드마크 일부를 찾지 못하는 등 규칙 실행 중 오류가 발생하면 실패 처리
            print(f"Error during pose analysis for '{keyword}': {e}")
            return False