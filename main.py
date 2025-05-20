from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routers import router
from predictor import EnhancedBTCPredictor
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Price Predictor Pro", version="4.3")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

predictor = EnhancedBTCPredictor()

@app.on_event("startup")
async def startup():
    predictor.initialize()

# Подключение маршрутов
app.include_router(router)
