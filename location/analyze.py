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

    candidates = ["Gwanghwamun", "Haebangchon", "Hangang Park", "Bukchon Hanok Village",
                  "Hongdae", "Itaewon", "Namsan Tower", "DDP", "Seoul Forest", "Lotte Tower",
                  "Gyeongbokgung Palace", "Deoksugung Palace", "Sewoon Arcade", "Euljiro", "Samcheong-dong",
                  "Cheonggyecheon", "Seongsu-dong", "Seoul National Univ. Station",
                  "Konkuk Univ. Station", "COEX"]

    result = classify_location(image, candidates)
    return JSONResponse(content=result)

