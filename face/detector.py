import mediapipe as mp
import numpy as np
import cv2

# Face Mesh 모델 초기화 (서버 시작 시 한번만 로드)
mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def analyze_face_expression(image_bgr: np.ndarray, keyword: str) -> bool:
    """MediaPipe Face Mesh를 사용하여 얼굴 표정을 분석합니다."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    if not results.multi_face_landmarks:
        print("⚠️ [analyze_face] 이미지에서 얼굴을 찾을 수 없습니다.")
        return False

    # 첫 번째로 감지된 얼굴의 랜드마크 사용
    face_landmarks = results.multi_face_landmarks[0].landmark

    # 얼굴 크기 정규화를 위해 얼굴 높이 계산
    # (턱 끝: 152, 이마 위쪽: 10)
    face_height = np.linalg.norm(
        np.array([face_landmarks[152].x, face_landmarks[152].y]) -
        np.array([face_landmarks[10].x, face_landmarks[10].y])
    )

    if keyword == "Smile":
        # 입꼬리가 올라갔는지, 입 너비가 넓어졌는지로 미소 판단
        # (오른쪽 입꼬리: 61, 왼쪽 입꼬리: 291)
        right_corner = np.array([face_landmarks[61].x, face_landmarks[61].y])
        left_corner = np.array([face_landmarks[291].x, face_landmarks[291].y])

        # 입 너비를 얼굴 높이로 정규화
        mouth_width = np.linalg.norm(right_corner - left_corner)
        normalized_mouth_width = mouth_width / face_height

        # 임계값은 실험을 통해 조정 필요
        if normalized_mouth_width > 0.4:
            return True

    elif keyword == "Surprised Face":
        # 입이 세로로 벌어지고, 눈이 커졌는지로 놀란 표정 판단
        # (윗입술 중앙: 13, 아랫입술 중앙: 14)
        upper_lip = np.array([face_landmarks[13].x, face_landmarks[13].y])
        lower_lip = np.array([face_landmarks[14].x, face_landmarks[14].y])

        # 입 높이를 얼굴 높이로 정규화
        mouth_height = np.linalg.norm(upper_lip - lower_lip)
        normalized_mouth_height = mouth_height / face_height

        # 임계값은 실험을 통해 조정 필요
        if normalized_mouth_height > 0.15:
            return True

    return False