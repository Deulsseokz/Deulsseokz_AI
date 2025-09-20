from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
from sqlalchemy import create_engine, text

# --- 1. DB 접속 설정 ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ConnectionError("Database environment variables are not fully set.")

DATABASE_URL = f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# --- 2. AI 모델 및 장소 목록 캐시 로드 ---
MODEL_NAME = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
print("✅ CLIP Model and Processor loaded.")

PLACES_CACHE = {} # {placeId: placeName} 형태의 캐시
CANDIDATES = {}   # {placeId: prompt} 형태의 후보
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT placeId, placeName, area FROM places_place"))
        for row in result:
            place_id, place_name, area = row[0], row[1], row[2]
            PLACES_CACHE[place_id] = place_name
            CANDIDATES[place_id] = f"A photo of {place_name} in {area}"
    print(f"✅ Successfully loaded {len(PLACES_CACHE)} places into cache.")
except Exception as e:
    print(f"🚨 FAILED to build candidates from database: {e}")

CANDIDATE_IDS = list(CANDIDATES.keys())
CANDIDATE_PROMPTS = list(CANDIDATES.values())


def get_place_name(place_id: int) -> str:
    """메모리에 저장된 캐시에서 placeId로 장소 이름을 찾음"""
    return PLACES_CACHE.get(place_id, "Unknown Place")


def classify_location(image: Image.Image):
    """주어진 이미지와 가장 일치하는 장소의 placeId와 확률을 반환"""
    if not CANDIDATES:
        return {"error": "Candidates not loaded. Check DB connection."}

    # 1. 이미지와 텍스트 프롬프트를 모델이 이해하도록 전처리
    inputs = processor(text=CANDIDATE_PROMPTS, images=image, return_tensors="pt", padding=True)

    # 2. 모델을 통해 이미지-텍스트 간 유사도 계산 (추론)
    # torch.no_grad()는 불필요한 연산을 막아 속도를 높여줒ㅁ
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. 계산된 결과(logits)를 확률(%)로 변환
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze()

    # 4. 가장 높은 확률을 가진 후보의 인덱스를 찾음
    best_candidate_index = probs.argmax().item()

    # 5. 인덱스를 이용해 placeId, 프롬프트, 확률 값을 가져옴
    best_place_id = CANDIDATE_IDS[best_candidate_index]
    best_prob = probs[best_candidate_index].item()
    best_prompt = CANDIDATE_PROMPTS[best_candidate_index]

    # 6. 결과를 딕셔너리 형태로 반환
    return {
        "best_match_place_id": best_place_id,
        "matched_prompt": best_prompt,
        "probability": best_prob
    }