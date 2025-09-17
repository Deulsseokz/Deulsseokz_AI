# location/clip_model.py

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import django

# --- Django 모델을 FastAPI(독립 스크립트)에서 사용하기 위한 설정 ---
# 아래 'config.settings'는 실제 프로젝트의 설정 파일 경로에 맞게 수정해야 합니다.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
# -------------------------------------------------------------

# Django 설정이 로드된 후에 모델을 임포트해야 합니다.
from places.models import Place

# --- 1. 서버 시작 시 모델을 한번만 로드 ---
MODEL_NAME = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
print("✅ CLIP Model and Processor loaded.")


# --- 2. DB에서 후보 목록을 동적으로 생성 ---
def build_candidates_from_db():
    """
    Place 테이블에서 모든 장소 정보를 가져와
    CLIP이 사용할 후보 딕셔너리를 생성합니다.
    """
    print("Building candidates from database...")
    candidates = {}
    places = Place.objects.all()
    for place in places:
        # 정확도를 높이기 위해 지역 정보를 포함한 프롬프트를 생성합니다.
        prompt = f"A photo of {place.placeName} in {place.area}"
        # placeId를 key로 사용합니다.
        candidates[place.placeId] = prompt

    print(f"✅ {len(candidates)} candidates built.")
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
        return {"error": "Candidates not loaded."}

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