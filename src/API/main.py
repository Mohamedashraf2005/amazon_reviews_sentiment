# uvicorn src.API.main:app --reload --port 8765
import os
from fastapi import FastAPI, HTTPException
import nltk
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
import numpy as np
import pickle
import re

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR) 

def get_path(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)

CHECKPOINT_PATH = get_path("models/checkpoint-59034")
BASE_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"  

with open(get_path("models/logistic__regression_model.pkl"), "rb") as f:
    logistic_model = pickle.load(f)

with open(get_path("models/xgboost_model.pkl"), "rb") as f:
    xgb_model = pickle.load(f)

with open(get_path("preprocessing/tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(get_path("preprocessing/scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

    
print("Classical models loaded ✓")

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

print("Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(f"Loading fine-tuned weights from:\n  {CHECKPOINT_PATH}")
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.eval()
print("Model ready ✓")

app = FastAPI(
    title="Amazon Review Classifier",
    description="Fine-tuned all-MiniLM-L6-v2 — 3-class sentiment (negative / neutral / positive)",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    texts: List[str]
    model: str   
    return_probabilities: bool = False   # set True to get full softmax scores

class Prediction(BaseModel):
    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: dict | None = None

class ClassifyResponse(BaseModel):
    predictions: List[Prediction]

def predict(texts: List[str], return_probs: bool = False) -> List[dict]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,          
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits     

    probs      = F.softmax(logits, dim=-1)      
    label_ids  = probs.argmax(dim=-1).tolist()  
    confidences = probs.max(dim=-1).values.tolist()

    results = []
    for i, text in enumerate(texts):
        entry = {
            "text":       text,
            "label":      ID2LABEL[label_ids[i]],
            "label_id":   label_ids[i],
            "confidence": round(confidences[i], 4),
        }
        if return_probs:
            entry["probabilities"] = {
                ID2LABEL[j]: round(probs[i][j].item(), 4)
                for j in range(3)
            }
        results.append(entry)

    return results


def cleaning(text):
    text = re.sub(r'https?:\/\/\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
    text=re.sub(r'<.*?>',' ',text)
    text=text.lower()
    text = text.split()
    text = ' '.join(text)
    return text

import string
punc=string.punctuation
def remove_punc(text):
    return text.translate(str.maketrans('','',punc))

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

important_words = {
    'not', 'no', 'nor', 'never',
    'very', 'too', 'so',
    'but', 'however'
}

custom_stopwords = stop_words - important_words

def remove_stopWords(words):
    return [word for word in words if word.lower() not in custom_stopwords]

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def lemmatizer_word(text):
 lemmas=[lemmatizer.lemmatize(word,pos='v') for word in text  ]
 return lemmas



def prepare_features(text):
    cleaned = cleaning(text)
    no_punc = remove_punc(cleaned)
    tokens = word_tokenize(no_punc)
    no_stop = remove_stopWords(tokens)
    lemmas = lemmatizer_word(no_stop)
    processed_text = " ".join(lemmas)

    text_tfidf = tfidf.transform([processed_text])

    text_len = len(text)
    words = text.split()
    word_cnt = len(words)
    lexical = len(set(words)) / word_cnt if word_cnt > 0 else 0
    caps = sum(1 for c in text if c.isupper()) / (text_len + 1)

    extra_features = np.array([[lexical, 0, text_len, caps]])
    extra_scaled = scaler.transform(extra_features)

    return hstack([text_tfidf, extra_scaled])



@app.get("/health")
def health():
    return {"status": "ok", "model": "checkpoint-59034", "classes": list(ID2LABEL.values())}

@app.post("/classify")
def classify(req: ClassifyRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")

    text = req.texts[0]

    if req.model == "minilm":
        result = predict([text], return_probs=True)[0]

    elif req.model == "logistic":
        X = prepare_features(text)
        probs = logistic_model.predict_proba(X)[0]
        label_id = probs.argmax()
        result = {
            "text": text,
            "label": ID2LABEL[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {
                ID2LABEL[i]: float(probs[i]) for i in range(3)
            }
        }

    elif req.model == "xgboost":
        X = prepare_features(text)
        probs = xgb_model.predict_proba(X)[0]
        label_id = probs.argmax()
        result = {
            "text": text,
            "label": ID2LABEL[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {
                ID2LABEL[i]: float(probs[i]) for i in range(3)
            }
        }

    else:
        raise HTTPException(status_code=400, detail="Invalid model")

    return {"predictions": [result]}