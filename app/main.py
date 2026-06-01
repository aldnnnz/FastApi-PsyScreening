from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

from app.schemas import PredictRequest
from app.predictor import run_late_fusion_prediction


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Mental Health API",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Too many requests. Please try again later."},
))


@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    return {
        "status": "ok"
    }


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: PredictRequest):
   
    tabular_dict = data.model_dump()

    user_text = tabular_dict.pop("text")

    result = run_late_fusion_prediction(
        user_text=user_text,
        tabular_dict=tabular_dict
    )

    return result