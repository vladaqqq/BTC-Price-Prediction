from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import router
import uvicorn

app = FastAPI(title="BTC Predictor Pro", version="5.3")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9331)