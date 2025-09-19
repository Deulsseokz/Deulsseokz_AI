import sys
import os
import io
import json
from typing import List, Dict, Callable, Any

# --- 1. Django 프로젝트 경로 설정 ---
# TODO: 이 경로가 Django 프로젝트의 루트 폴더(manage.py가 있는 곳)가 맞는지 다시 한번 확인해주세요.
# '/Users/minkyoungshin/desktop/Deulsseokz_BE' 가 더 정확한 경로일 수 있습니다.
DJANGO_PROJECT_PATH = '/Users/minkyoungshin/desktop/Deulsseokz_BE/Deulsseokz_BE'
sys.path.append(DJANGO_PROJECT_PATH)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

# --- 2. 모든 import 구문을 파일 상단으로 이동 ---
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2

from places.models import Place
from location.clip_model import classify_location
from pose.detector import analyze_strong_pose

# --- 3. .env 파일 로드 및 환경 변수 설정 ---
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 분석 종류별 처리 함수 정의 ---
def check_location(image_pil: Image.Image, expected_place_name: str, **kwargs) -> Dict[str, Any]:
    """CLIP 모델로 장소를 분석하고, 기대하는 장소와 일치하는지 확인합니다."""
    # ✅ from places.models import Place 구문을 파일 상단으로 이동시켰습니다.
    try:
        classification_result = classify_location(image_pil)
        best_match_id = classification_result.get("best_match_place_id")
        probability = classification_result.get("probability", 0)

        matched_place = Place.objects.get(placeId=best_match_id)

        print(f"Location Check | Expected: {expected_place_name}, AI Predicted: {matched_place.placeName}, Prob: {probability:.2f}")

        if matched_place.placeName == expected_place_name and probability > 0.7:
            return {"success": True}
        else:
            return {"success": False}
    except Exception as e:
        print(f"Error during location check: {e}")
        return {"success": False}

# ... 이하 check_strong_pose, ANALYSIS_DISPATCHER, analyze_challenge 함수는 기존과 동일 ...
def check_strong_pose(image_bgr: np.ndarray, **kwargs) -> Dict[str, Any]:
    """MediaPipe 모델로 'StrongPose'를 분석합니다."""
    try:
        is_correct = analyze_strong_pose(image_bgr)
        print(f"StrongPose Check | Result: {is_correct}")
        return {"success": is_correct}
    except Exception as e:
        print(f"Error during StrongPose check: {e}")
        return {"success": False}


ANALYSIS_DISPATCHER: Dict[str, Callable] = {
    "PalaceGate": check_location,
    "StrongPose": check_strong_pose,
}

@app.post("/analyze")
async def analyze_challenge(
        image: UploadFile = File(...),
        conditions: str = Form(...),
        place_name: str = Form(...)
):
    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
            result_dict = analysis_function(
                image_pil=image_pil,
                image_bgr=image_bgr,
                expected_place_name=place_name
            )
            is_condition_met = result_dict.get("success", False)

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