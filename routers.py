from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from predictor import BTCPredictor

templates = Jinja2Templates(directory="templates")
router = APIRouter()
predictor = BTCPredictor()


@router.on_event("startup")
async def startup_event():
    predictor.initialize()


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "initialized": predictor.initialized,
            "metrics": predictor.metrics
        }
    )


@router.get("/api/predict")
async def get_predict(days: str = "1,3,7,30,90"):
    try:
        if not predictor.initialized:
            return {"status": "error", "error": "Model not initialized"}
        return {"status": "success", **predictor.predict([int(d) for d in days.split(',')])}
    except Exception as e:
        return {"status": "error", "error": str(e)}
