from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from test import *

load_dotenv()
MODEL_API = os.getenv("MODEL_API")

app = FastAPI(title="Embryo Multi-Task Prediction")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FRONTEND_API"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = load_model()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        x = preprocess_image(image_bytes)
        seg_mask, cls_pred = model.predict(x)
        
        cls_label = postprocess_classification(cls_pred)

        return JSONResponse({
            "masks": seg_mask.tolist() if hasattr(seg_mask, "tolist") else seg_mask,
            "classification": (
                cls_label.tolist() if hasattr(cls_label, "tolist") else cls_label
            ),
        })
    except Exception as e:
        print("Prediction error:", e) 
        return JSONResponse({"error": str(e)}, status_code=500)