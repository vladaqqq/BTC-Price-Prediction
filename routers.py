from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from predictor import predictor
from main import templates
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/api/predict")
async def predict(days: str = "1,3,7,30,90,180"):
    try:
        days_list = [int(d) for d in days.split(",")]
        df = predictor.load_data(days=60)
        df = predictor.enhanced_feature_engineering(df)
        X = df.drop(columns=['target'], errors='ignore').tail(1)
        result = predictor.predict_multiple_days(X, days_list)

        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"status": "error", "error": str(e)}
