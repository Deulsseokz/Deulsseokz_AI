from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2
import io
import json
from typing import List, Dict, Callable, Any
import os
import django

# --- 1. Django 모델을 FastAPI에서 사용하기 위한 설정 ---
# Django 프로젝트의 settings.py 경로를 정확하게 지정해야 합니다.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from places.models import Place
# -------------------------------------------------------------


# --- 2. 각 폴더의 분석 함수들을 import 합니다. ---
from location.clip_model import classify_location  # 장소 분석 함수

# from hands.detector import detect_hand_pose      # TODO: MediaPipe 손 포즈 함수
# from pose.detector import detect_body_pose       # TODO: MediaPipe 몸 포즈 함수


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # ... (기존과 동일) ...
)


# --- 3. 분석 함수들을 정의하고, 키워드와 짝지어 줍니다. ---

def check_location_match(image_pil: Image.Image, expected_place_name: str, **kwargs) -> Dict[str, Any]:
    """CLIP 모델로 장소를 분석하고, 기대하는 장소와 일치하는지 확인합니다."""
    try:
        # clip_model.py의 함수를 호출하여 AI가 판단한 가장 유사한 장소의 ID를 받음
        classification_result = classify_location(image_pil)
        best_match_id = classification_result.get("best_match_place_id")
        probability = classification_result.get("probability", 0)

        # AI가 판단한 장소의 이름을 DB에서 조회
        matched_place = Place.objects.get(placeId=best_match_id)

        print(
            f"Expected: {expected_place_name}, AI Predicted: {matched_place.placeName}, Probability: {probability:.2f}")

        # 기대 장소와 AI 판단 장소가 일치하고, 확률이 70% 이상이면 성공
        if matched_place.placeName == expected_place_name and probability > 0.7:
            return {"success": True}
        else:
            return {"success": False}

    except Exception as e:
        print(f"Error during location check: {e}")
        return {"success": False}


def check_pose_placeholder(image_bgr: np.ndarray, keyword: str, **kwargs) -> Dict[str, Any]:
    """MediaPipe 포즈 분석을 위한 임시 함수입니다."""
    print(f"Running placeholder pose analysis for: {keyword}")
    # TODO: 여기에 실제 MediaPipe 분석 함수(detect_body_pose 등)를 연결하세요.
    return {"success": True}  # 임시로 항상 성공 반환


# 키워드와 실제 분석 함수를 연결하는 '전화번호부'
ANALYSIS_DISPATCHER: Dict[str, Callable] = {
    # 장소 관련 키워드는 모두 check_location_match 함수로 연결
    "PalaceGate": check_location_match,
    "Landmark": check_location_match,
    "RiverView": check_location_match,
    # TODO: 포즈 관련 키워드들을 MediaPipe 함수로 연결
    "StrongPose": check_pose_placeholder,
    "PeaceSign": check_pose_placeholder,
}


@app.post("/analyze")
async def analyze_challenge(
        image: UploadFile = File(...),
        conditions: str = Form(...),
        place_name: str = Form(...)  # ✅ 1단계에서 추가한 장소 이름을 받음
):
    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.IOBase(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        conditions_list: List[str] = json.loads(conditions)
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid input: {e}"})

    details_result = {}
    condition_results = []

    for i, keyword in enumerate(conditions_list, 1):
        analysis_function = ANALYSIS_DISPATCHER.get(keyword)
        is_condition_met = False

        if analysis_function:
            try:
                # 함수에 필요한 인자들을 전달
                result_dict = analysis_function(
                    image_pil=image_pil,
                    image_bgr=image_bgr,
                    keyword=keyword,
                    expected_place_name=place_name
                )
                is_condition_met = result_dict.get("success", False)
            except Exception as e:
                print(f"Error during '{keyword}' analysis: {e}")
        else:
            print(f"Warning: No analysis function found for keyword '{keyword}'")

        details_result[f"condition{i}_met"] = is_condition_met
        condition_results.append(is_condition_met)

    overall_success = all(condition_results) if condition_results else False
    message = "챌린지 모든 조건 충족!" if overall_success else "일부 조건을 만족하지 못했습니다."

    final_response = {
        "success": overall_success,
        "message": message,
        "details": details_result
    }

    return JSONResponse(content=final_response)