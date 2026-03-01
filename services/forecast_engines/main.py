from pydantic import BaseModel
from fastapi import FastAPI
from typing import List

app = FastAPI()

class ForecastRequest(BaseModel):
    target: str
    neighbors: List[str]
    horizon: int


@app.post("/forecast")
def forecast(req: ForecastRequest):
    return {
        "target": req.target,
        "neighbors": req.neighbors,
        "horizon": req.horizon,
        "mae": 0.0
    }