import io
import json
import os
from typing import List, Dict, Callable, Any

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text

# --- 1. .env 파일 로드 ---
load_dotenv()

# --- 2. DB 접속 설정 ---
# Docker 컨테이너 실행 시 전달된 환경 변수를 사용합니다.
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    print("🚨 WARNING: Database environment variables are not fully set. Using placeholders.")
    # raise ConnectionError("Database environment variables are not fully set.")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# --- 3. 각 모듈의 실제 분석 함수들을 import ---
# (TODO: 아래 함수 이름들은 실제 작성하신 함수명으로 정확히 수정해야 합니다)
from location.clip_model import classify_location  # CLIP 장소 분석 함수
from pose.detector import analyze_strong_pose   # MediaPipe "StrongPose" 분석 함수

# --- 4. 서버 시작 시 DB에서 장소 목록을 미리 로드하여 메모리에 저장 (캐싱) ---
PLACES_CACHE = {}  # {placeId: placeName} 형태의 캐시
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT placeId, placeName FROM places_place"))
        for row in result:
            PLACES_CACHE[row[0]] = row[1]  # row[0]은 placeId, row[1]은 placeName
    print(f"✅ Successfully loaded {len(PLACES_CACHE)} places from the database into cache.")
except Exception as e:
    print(f"🚨 FAILED to connect to the database and build cache: {e}")

# --- FastAPI 앱 생성 및 미들웨어 설정 ---
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
    try:
        classification_result = classify_location(image_pil)
        best_match_id = classification_result.get("best_match_place_id")
        probability = classification_result.get("probability", 0)

        # 미리 로드한 메모리 캐시에서 장소 이름을 찾음
        matched_place_name = PLACES_CACHE.get(best_match_id, "Unknown Place")

        print(
            f"Location Check | Expected: {expected_place_name}, AI Predicted: {matched_place_name}, Prob: {probability:.2f}")

        if matched_place_name == expected_place_name and probability > 0.7:
            return {"success": True}
        else:
            return {"success": False}
    except Exception as e:
        print(f"Error during location check: {e}")
        return {"success": False}

def check_strong_pose(image_bgr: np.ndarray, **kwargs) -> Dict[str, Any]:
    """MediaPipe 모델로 'StrongPose'를 분석"""
    try:
        is_correct = analyze_strong_pose(image_bgr)
        print(f"StrongPose Check | Result: {is_correct}")
        return {"success": is_correct}
    except Exception as e:
        print(f"Error during StrongPose check: {e}")
        return {"success": False}

# --- 조건 키워드와 분석 함수를 연결하는 '라우터' ---
ANALYSIS_DISPATCHER: Dict[str, Callable] = {
    "PalaceGate": check_location,
    "StrongPose": check_strong_pose,
    # TODO: 다른 조건 키워드와 함수를 여기에 추가
}

@app.post("/analyze")
async def analyze_challenge(
        image: UploadFile = File(...),
        conditions: str = Form(...),
        place_name: str = Form(...)
):
    """Django로부터 챌린지 분석 요청을 받아 처리하는 메인 엔드포인트"""
    try:
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB_BGR)
        conditions_list: List[str] = json.loads(conditions)
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid input: {e}"})

    details_result = {}
    condition_results = []

    # 전달받은 모든 조건에 대해 순서대로 분석 실행
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

    # 모든 조건이 True여야만 최종 성공
    overall_success = all(condition_results) if condition_results else False
    message = "챌린지 모든 조건 충족!" if overall_success else "일부 조건을 만족하지 못했습니다."

    # Django와 약속된 최종 결과 형식으로 반환
    final_response = {
        "success": overall_success,
        "message": message,
        "details": details_result
    }

    return JSONResponse(content=final_response)