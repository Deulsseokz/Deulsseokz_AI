from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from hands.detector import detect_pose
from location.analyze import router as location_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/pose")
async def analyze_pose(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result = detect_pose(image_bgr)
    return JSONResponse(content=result)

app.include_router(location_router)
# app.include_router(analyze_router)