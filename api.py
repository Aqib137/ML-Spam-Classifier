from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model_obj = joblib.load(MODEL_PATH)
pipeline = model_obj["pipeline"]
encoder = model_obj["label_encoder"]

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Classifier API is running"}

@app.post("/predict")
def predict(data: TextInput):
    

    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    text = [data.text]
    
    pred = pipeline.predict(text)[0]
    probs = pipeline.predict_proba(text)[0]

    label = encoder.inverse_transform([pred])[0]
    confidence = probs[pred]

    return {
        "input": data.text,
        "prediction": label,
        "confidence": float(confidence)
    }
