# ML Spam Classifier

A small end-to-end spam classifier project that trains a model based on the SMS Spam Collection dataset.

## Structure

- `data/` - dataset files (downloaded automatically)
- `model/` - trained model artifact (`model.pkl`)
- `train.py` - trains the model and evaluates performance
- `predict.py` - loads the model and predicts spam/ham for new text

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

### 3) Make predictions

```bash
python predict.py --text "Free entry in 2 a wkly comp to win FA Cup final tkts"
```

Or predict in batch:

```bash
python predict.py --file some_texts.txt
```
