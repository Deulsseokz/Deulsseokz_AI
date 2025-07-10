from fastapi import FastAPI, UploadFile, File
from pose import detect_pose_and_crop_hands
from hands import detect_v_sign
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()

@app.post("/analyze/pose-v")
async def analyze_pose_and_v_sign(image: UploadFile = File(...)):
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    hands_crops = detect_pose_and_crop_hands(img)
    result = {"v_sign_detected": False, "details": []}

    for hand_img in hands_crops:
        is_v, info = detect_v_sign(hand_img)
        result["details"].append(info)
        if is_v:
            result["v_sign_detected"] = True

    return JSONResponse(content=result)
