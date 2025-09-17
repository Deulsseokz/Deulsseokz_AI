# location/analyze.py

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

# clip_model 모듈에서 실제 분석 함수를 가져옵니다.
from location.clip_model import classify_location

router = APIRouter()


@router.post("/analyze/location")
async def analyze_location(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # clip_model.py에 미리 준비된 CANDIDATES를 사용하여 분류 수행
    result = classify_location(image)

    return JSONResponse(content=result)