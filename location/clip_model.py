# location/clip_model.py (서버 분리 환경 최종본)

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
from sqlalchemy import create_engine, text

# --- Django 직접 의존성 제거 ---
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
# django.setup()
# from places.models import Place
# --------------------------------

# --- 1. DB 접속 설정 (main.py와 동일) ---
# Docker 컨테이너 실행 시 전달된 환경 변수를 사용합니다.
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

# DB 접속 정보가 하나라도 없으면 에러 발생 (안전장치)
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ConnectionError("Database environment variables are not fully set.")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# --- 2. 서버 시작 시 모델을 한번만 로드 ---
MODEL_NAME = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
print("✅ CLIP Model and Processor loaded.")


# --- 3. DB에서 후보 목록을 동적으로 생성 ---
def build_candidates_from_db():
    """
    places_place 테이블에 직접 접속하여 모든 장소 정보를 가져와
    CLIP이 사용할 후보 딕셔너리를 생성합니다.
    """
    print("Building candidates from database...")
    candidates = {}
    try:
        with engine.connect() as connection:
            # ✅ Django ORM 대신 SQLAlchemy로 직접 쿼리
            result = connection.execute(text("SELECT placeId, placeName, area FROM places_place"))
            for row in result:
                # row[0]: placeId, row[1]: placeName, row[2]: area
                prompt = f"A photo of {row[1]} in {row[2]}"
                candidates[row[0]] = prompt
        print(f"✅ {len(candidates)} candidates built.")
    except Exception as e:
        print(f"🚨 FAILED to build candidates from database: {e}")

    return candidates


# 서버가 시작될 때 후보 목록을 한번만 생성하여 메모리에 저장
CANDIDATES = build_candidates_from_db()
CANDIDATE_IDS = list(CANDIDATES.keys())
CANDIDATE_PROMPTS = list(CANDIDATES.values())


def classify_location(image: Image.Image):
    """
    주어진 이미지와 DB에서 불러온 장소 후보들을 비교하여,
    가장 일치하는 장소의 placeId와 확률을 반환합니다.
    """
    if not CANDIDATES:
        return {"error": "Candidates not loaded. Check DB connection."}

    # 이미지와 텍스트 프롬프트를 모델이 이해하도록 전처리
    inputs = processor(text=CANDIDATE_PROMPTS, images=image, return_tensors="pt", padding=True)

    # 모델을 통해 유사도 계산
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze()

    # 가장 높은 확률을 가진 후보의 인덱스를 찾음
    best_candidate_index = probs.argmax().item()

    # 인덱스를 이용해 placeId와 확률을 가져옴
    best_place_id = CANDIDATE_IDS[best_candidate_index]
    best_prob = probs[best_candidate_index].item()
    best_prompt = CANDIDATE_PROMPTS[best_candidate_index]

    return {
        "best_match_place_id": best_place_id,
        "matched_prompt": best_prompt,
        "probability": best_prob
    }