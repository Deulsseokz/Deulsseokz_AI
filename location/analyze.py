from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from location.clip_model import classify_location

router = APIRouter()

@router.post("/analyze/location")
async def analyze_location(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 후보 장소 딕셔너리 (key: 장소명, value: 설명)
    candidates = {
        # "Gwanghwamun": "광화문 앞 광장의 모습",
        # "Haebangchon": "해방촌 거리 풍경",
        # "Hangang Park": "한강 공원의 자전거 도로",
        # "Bukchon Hanok Village": "북촌 한옥마을의 전통 가옥",
        # "Hongdae": "홍대 거리의 예술 벽화",
        # "Itaewon": "이태원의 밤거리",
        # "Namsan Tower": "남산 타워가 보이는 도시 풍경",
        # "DDP": "동대문디자인플라자(DDP)의 곡선 건물",
        # "Seoul Forest": "서울숲의 녹음진 공원",
        # "Lotte Tower": "롯데타워의 초고층 건물",
        # "Gyeongbokgung Palace": "경복궁의 전통 궁궐 건물",
        # "Deoksugung Palace": "덕수궁 돌담길",
        # "Sewoon Arcade": "세운상가의 복합 건물",
        # "Euljiro": "을지로의 오래된 골목",
        # "Samcheong-dong": "삼청동의 카페 거리",
        # "Cheonggyecheon": "청계천의 도심 하천",
        # "Seongsu-dong": "성수동의 공장 리모델링 거리",
        # "Seoul National Univ. Station": "서울대입구역 근처 풍경",
        # "Konkuk Univ. Station": "건대입구의 번화가",
        "COEX Starfield Library": "COEX Starfield Library with tall curved bookshelves and warm lighting"
    }

    result = classify_location(image, candidates)
    return JSONResponse(content=result)
