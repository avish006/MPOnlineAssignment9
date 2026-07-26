from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ML Model Deployment API", version="1.0")

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float

class PredictionResponse(BaseModel):
    prediction: float

# A dummy model for demonstration purposes
def dummy_predict(f1, f2):
    return (f1 + f2) * 2.5

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API! Send a POST request to /predict to get a prediction."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = dummy_predict(request.feature1, request.feature2)
    return PredictionResponse(prediction=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
