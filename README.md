# Spam Detection API (FastAPI + ML)

A production-ready NLP API that classifies SMS/text messages as spam or not spam in real-time.

- 🚀 Deployed API: https://ml-spam-classifier-a2s1.onrender.com/docs
- 🧠 Model: TF-IDF + Logistic Regression
- 📊 Performance: ~93% F1-score on spam detection

## Features

- Real-time spam detection via REST API
- Robust text preprocessing using TF-IDF
- Model served with FastAPI
- Deployed on cloud (Render)
- Handles edge cases (empty input, noisy text)

## API Usage

POST /predict

Request:
```json
{
  "text": "Win a free iPhone now!!!"
}
```

Response:
```json
{
  "prediction": "spam",
  "confidence": 0.94
}
```

## Real-World Use Cases

- Email/SMS spam filtering systems
- Fraud/scam detection
- Content moderation pipelines

## Structure

- `data/` - dataset files (downloaded automatically)
- `model/` - trained model artifact (`model.pkl`)
- `train.py` - trains the model and evaluates performance
- `predict.py` - loads the model and predicts spam/ham for new text
- `api.py` - FastAPI server for predictions

## Quick Start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Train the model

```bash
python train.py
```

Output will include evaluation metrics and will write `model/model.pkl`.

### 3) Make predictions locally

```bash
python predict.py --text "Free entry in 2 a wkly comp to win FA Cup final tkts"
```

Or predict in batch:

```bash
python predict.py --file some_texts.txt
```

### 4) Run the API locally

```bash
uvicorn api:app --reload
```

Visit http://127.0.0.1:8000/docs for interactive API docs.
