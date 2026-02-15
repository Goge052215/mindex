"""
main pipeline for sentiment analysis
    - load model
    - predict sentiment
    - return sentiment and probability distribution

@goge052215
"""

import sys
import os
import pickle
import json
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'sentiment_model.pkl')
_MODEL = None

def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        _MODEL = pickle.load(f)
    return _MODEL

def predict(text_or_texts):
    model = load_model()
    if model is None:
        return {"status": "error", "message": "Model not found. Please run src/train.py first."}
    try:
        sentiment_map = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}
        if isinstance(text_or_texts, list):
            predictions = model.predict(text_or_texts)
            probabilities = model.predict_proba(text_or_texts)
            data = []
            for text, prediction_code, probs in zip(text_or_texts, predictions, probabilities):
                sentiment = sentiment_map.get(prediction_code, "Unknown")
                probs_dict = {}
                for idx, class_label in enumerate(model.classes_):
                    label_name = sentiment_map.get(class_label, str(class_label))
                    probs_dict[label_name] = round(probs[idx] * 100, 1)
                data.append({
                    "text": text,
                    "sentiment": sentiment,
                    "distribution": probs_dict
                })
            return {
                "status": "success",
                "data": data
            }
        prediction_code = model.predict([text_or_texts])[0]
        probabilities = model.predict_proba([text_or_texts])[0]
        sentiment = sentiment_map.get(prediction_code, "Unknown")
        probs_dict = {}
        for idx, class_label in enumerate(model.classes_):
            label_name = sentiment_map.get(class_label, str(class_label))
            probs_dict[label_name] = round(probabilities[idx] * 100, 1)
        return {
            "status": "success", 
            "data": sentiment,
            "distribution": probs_dict
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error", 
            "message": "No input text provided"
        }))
        sys.exit(1)
        
    raw_text = sys.argv[1]
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            text_input = parsed
        else:
            text_input = raw_text
    except Exception:
        text_input = raw_text
    result = predict(text_input)
    print(json.dumps(result))
