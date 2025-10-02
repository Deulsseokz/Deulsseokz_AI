import io
import json
import os
from typing import List, Dict, Callable, Any

import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)


# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyze_api")

# --- 1. .env 파일 로드 ---
load_dotenv()

# --- 2. DB 접속 및 캐시 로드 코드를 모두 삭제 ---
# 이 작업은 이제 clip_model.py가 전담합니다.

# --- 3. 분석 모듈 import ---
from location.clip_model import classify_location, get_place_name, classify_attributes, processor, model
from pose.detector import analyze_pose
from hands.detector import analyze_hand_gesture
from face.detector import analyze_face_expression

# --- 4. FastAPI 초기화 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. 분석 함수 정의 ---
def check_location(image_pil: Image.Image, expected_place_name: str, **kwargs) -> Dict[str, Any]:
    try:
        classification_result = classify_location(image_pil)
        best_match_id = classification_result.get("best_match_place_id")
        probability = classification_result.get("probability", 0.0)

        # clip_model.py에 있는 get_place_name 함수를 사용하여 장소 이름을 가져옴
        matched_place_name = get_place_name(best_match_id)

        logger.info(
            f"[check_location] Expected: {expected_place_name}, Predicted: {matched_place_name}, Prob: {probability:.2f}")

        return {"success": matched_place_name == expected_place_name and probability > 0.5}

    except Exception as e:
        logger.error(f"[check_location] Error: {e}")
        return {"success": False}


def classify_attributes(image: Image.Image, keywords: List[str]) -> dict:
    """
    주어진 이미지와 임의의 키워드 목록을 비교하여,
    이미지와 가장 잘 맞는 키워드와 그 확률을 반환
    """
    if not keywords:
        return {"error": "No keywords provided."}

    # 키워드를 모델이 이해하기 좋은 프롬프트로 변환
    prompts = [f"a photo of a {k.lower().replace('_', ' ')}" for k in keywords]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze()

    best_index = probs.argmax().item()
    best_keyword = keywords[best_index]
    best_prob = probs[best_index].item()

    return {
        "best_match_keyword": best_keyword,
        "probability": best_prob
    }

def check_attribute(image_pil: Image.Image, keyword: str, **kwargs) -> dict:
    """이미지에 특정 속성(키워드)이 있는지 판단"""
    try:
        result = classify_attributes(image_pil, [keyword])
        probability = result.get("probability", 0.0)
        logger.info(f"[check_attribute] Keyword: {keyword}, Prob: {probability:.2f}")
        return {"success": probability > 0.5}
    except Exception as e:
        logger.error(f"[check_attribute] Error for keyword '{keyword}': {e}")
        return {"success": False}

def check_pose(image_bgr: np.ndarray, keyword: str, **kwargs) -> dict:
    """MediaPipe 모델로 이미지에서 특정 포즈를 분석."""
    try:
        is_correct = analyze_pose(image_bgr, keyword)
        logger.info(f"[check_pose] Keyword: {keyword}, Result: {is_correct}")
        return {"success": is_correct}
    except Exception as e:
        logger.error(f"[check_pose] Error for '{keyword}': {e}")
        return {"success": False}

def check_hand_gesture(image_bgr: np.ndarray, keyword: str, **kwargs) -> dict:
    """손 포즈 분석을 호출"""
    try:
        is_correct = analyze_hand_gesture(image_bgr, keyword)
        logger.info(f"[check_hand_gesture] Keyword: {keyword}, Result: {is_correct}")
        return {"success": is_correct}
    except Exception as e:
        logger.error(f"[check_hand_gesture] Error for '{keyword}': {e}")
        return {"success": False}

def check_face_expression(image_bgr: np.ndarray, keyword: str, **kwargs) -> dict:
    """얼굴 표정 분석을 호출"""
    try:
        is_correct = analyze_face_expression(image_bgr, keyword)
        logger.info(f"[check_face_expression] Keyword: {keyword}, Result: {is_correct}")
        return {"success": is_correct}
    except Exception as e:
        logger.error(f"[check_face_expression] Error for '{keyword}': {e}")
        return {"success": False}

# --- 6. 조건 매핑 ---
ANALYSIS_DISPATCHER: Dict[str, Callable] = {
    # --- 장소 이름 비교 ---
    "Landmark": check_location, "PalaceGate": check_location, "Temple": check_location,
    "Building": check_location, "GardenSign": check_location, "Seoul Station": check_location,

    # --- 일반 속성 판단 ---
    "RiverView": check_attribute, "SeaView": check_attribute, "LakeView": check_attribute,
    "Park": check_attribute, "Forest": check_attribute, "Field": check_attribute,
    "Hanok": check_attribute, "Bridge": check_attribute, "Cave": check_attribute,
    "Waterfall": check_attribute, "Street": check_attribute, "Mural": check_attribute,
    "Tree": check_attribute, "FlowerField": check_attribute, "RockView": check_attribute,

    # --- 포즈 판단 ---
    "StrongPose": check_pose, "Sitting": check_pose,
    "CreativePose": check_pose, "Deep Bow": check_pose, "Walking": check_pose,
    "Arms Up": check_pose, "Shielding Eyes": check_pose, "Back View": check_pose,
    "Crossing Arms": check_pose, "Spreading Arms": check_pose,

    # --- 손 판단 ---
    "PeaceSign": check_hand_gesture, "HeartPose": check_hand_gesture,
    "Point": check_hand_gesture, "HandsTogether": check_hand_gesture, "Flower Cup": check_hand_gesture,

    # --- 표정 판단 ---
    "Smile": check_face_expression,
    "Surprised Face": check_face_expression,
}

# --- 7. 메인 분석 엔드포인트 ---
@app.post("/analyze")
async def analyze_challenge(
    image: UploadFile = File(...),
    conditions: str = Form(...),
    place_name: str = Form(...)
):
    logger.info("📥 /analyze 호출됨")
    logger.info(f"📥 image.filename: {image.filename}")
    logger.info(f"📥 place_name: {place_name}")
    logger.info(f"📥 conditions: {conditions}")

    try:
        image_bytes = await image.read()
        logger.info(f"🖼️ image 크기: {len(image_bytes)} bytes")

        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        try:
            conditions_list: List[str] = json.loads(conditions)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON 파싱 실패: {e}")
            return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid JSON in 'conditions': {e}"})

    except Exception as e:
        logger.error(f"❌ 입력 처리 실패: {e}")
        return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid input: {e}"})

    details_result = {}
    condition_results = []

    logger.info("🚀 조건별 분석 시작")
    for i, keyword in enumerate(conditions_list, 1):
        logger.info(f"🔍 조건 {i}: {keyword}")
        analysis_function = ANALYSIS_DISPATCHER.get(keyword)
        is_condition_met = False

        if analysis_function:
            try:
                result_dict = analysis_function(
                    image_pil=image_pil,
                    image_bgr=image_bgr,
                    expected_place_name=place_name,
                    keyword=keyword
                )
                is_condition_met = result_dict.get("success", False)
                logger.info(f"✅ 조건 {i} 결과: {is_condition_met}")
            except Exception as e:
                logger.error(f"❌ 조건 {i} 예외: {e}")
        else:
            logger.warning(f"⚠️ 알 수 없는 조건: {keyword}")

        details_result[f"condition{i}_met"] = is_condition_met
        condition_results.append(is_condition_met)

    overall_success = all(condition_results)
    message = "챌린지 모든 조건 충족!" if overall_success else "일부 조건을 만족하지 못했습니다."
    logger.info(f"🎯 최종 결과: {overall_success} | 메시지: {message}")

    return JSONResponse(content={
        "success": overall_success,
        "message": message,
        "details": details_result
    })
