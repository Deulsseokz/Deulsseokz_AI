import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 관련 도구 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """세 점 사이의 각도를 계산하는 함수"""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def analyze_pose(image_bgr: np.ndarray, keyword: str) -> bool:
    """
    주어진 이미지와 키워드를 바탕으로 MediaPipe를 사용하여 포즈를 분석합니다.
    """
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # MediaPipe는 RGB 이미지를 사용하므로 변환
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # 포즈 검출 수행
        results = pose.process(image_rgb)

        # 랜드마크 추출
        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            print("Warning: No pose landmarks detected in the image.")
            return False  # 이미지에서 사람을 찾지 못하면 실패

        # --- 키워드별 포즈 규칙 정의 ---

        if keyword == "StrongPose":
            # '두 팔을 허리에 짚는 포즈' 규칙
            try:
                # 1. 양쪽 어깨, 팔꿈치, 손목, 엉덩이 좌표 가져오기
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # 2. 양쪽 팔꿈치 각도 계산 (약 90도 근처인지 확인)
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

                # 3. 양쪽 손목이 엉덩이 근처에 있는지 확인 (y좌표 기준)
                # (손목이 엉덩이보다 약간 위, 어깨보다는 아래에 위치)
                is_hand_on_hip_l = hip_l[1] > wrist_l[1] > shoulder_l[1]
                is_hand_on_hip_r = hip_r[1] > wrist_r[1] > shoulder_r[1]

                # 4. 최종 판정: 양쪽 팔꿈치 각도가 60~130도 사이이고, 양손이 엉덩이 근처에 있다면 성공
                if (60 < angle_l < 130 and 60 < angle_r < 130) and (is_hand_on_hip_l and is_hand_on_hip_r):
                    return True
                else:
                    return False
            except:
                return False  # 랜드마크 일부가 감지되지 않으면 실패

        elif keyword == "Sitting":
            try:
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)
                return 80 < angle < 140  # 앉아있을 때 무릎 각도

            except:
                return False


        elif keyword == "Jump":
            try:
                left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y

                # 발목이 코보다 위에 있으면 점프하는 것으로 간주 (단순한 규칙)
                return left_ankle_y < nose_y and right_ankle_y < nose_y

            except:
                return False

        else:
            # 정의되지 않은 포즈 키워드는 일단 성공 처리 (또는 False 처리)
            print(f"Unknown pose keyword: {keyword}")
            return True

    return False