"""Classify complaint component groups; never imply confirmed mechanical diagnosis."""
from functools import lru_cache
from backend.app.config import settings

CATEGORY_KEYWORDS = {
    'Engine / Powertrain': ('engine', 'stall', 'power', 'accelerat', 'misfire', 'fuel', 'rpm'),
    'Brakes': ('brak', 'abs', 'stop', 'pedal'),
    'Electrical': ('battery', 'alternator', 'electri', 'light', 'wiring'),
    'Steering / Suspension': ('steer', 'suspension', 'wheel', 'vibrat'),
    'Airbags / Safety': ('airbag', 'seatbelt', 'restraint'),
    'Transmission': ('transmission', 'gear', 'shift', 'clutch'),
}


@lru_cache(maxsize=1)
def _load_transformer():
    if not settings.transformer_dir.joinpath('config.json').exists():
        return None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(str(settings.transformer_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(settings.transformer_dir), local_files_only=True)
    model.to('cpu').eval()
    return tokenizer, model




@lru_cache(maxsize=1)
def _load_multilabel_v3():
    """Lazy-load CPU model, tokenizer, and validation-selected decision threshold."""
    model_dir = settings.multilabel_v3_dir
    required = (model_dir / 'config.json', model_dir / 'metrics.json')
    weights = any((model_dir / name).is_file()
                  for name in ('model.safetensors', 'pytorch_model.bin'))
    if not all(path.is_file() for path in required) or not weights:
        return None
    import json
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    metrics = json.loads((model_dir / 'metrics.json').read_text(encoding='utf-8'))
    if metrics.get('taxonomy') != 'v3' or metrics.get('architecture') != 'DistilBERT':
        raise RuntimeError('Expected V3 multi-label DistilBERT metrics and taxonomy')
    threshold = float(metrics['selected_threshold'])
    if not 0.0 < threshold < 1.0:
        raise ValueError('Invalid V3 classification threshold')
    max_length = int(metrics['max_length'])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    if model.config.problem_type != 'multi_label_classification':
        raise RuntimeError('V3 model is not configured for multi-label classification')
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if labels != metrics['labels']:
        raise RuntimeError('Model labels do not match V3 training metrics')
    model.to('cpu').eval()
    return tokenizer, model, threshold, max_length


def _predict_multilabel_v3(text: str) -> dict:
    bundle = _load_multilabel_v3()
    if bundle is None:
        return {'category': 'Model not installed', 'confidence': None,
                'labels': [], 'method': 'multilabel_v3_artifact_missing'}

    import torch
    tokenizer, model, threshold, max_length = bundle
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        scores = torch.sigmoid(logits).cpu().tolist()  # NOT softmax: labels can co-occur.
    ranked = sorted(zip((model.config.id2label[i] for i in range(len(scores))), scores),
                    key=lambda item: item[1], reverse=True)
    selected = [{'category': category, 'score': round(float(score), 4)}
                for category, score in ranked if score >= threshold]
    return {
        'category': selected[0]['category'] if selected else 'Uncertain',
        'confidence': selected[0]['score'] if selected else None,
        'labels': selected,
        'method': 'finetuned_multilabel_distilbert_v3',
    }


@lru_cache(maxsize=1)
def _load_baseline():
    if settings.model_file.exists():
        import joblib
        return joblib.load(settings.model_file)
    return None


def predict_category(text: str) -> dict:
    if settings.model_variant == 'multilabel_distilbert_v3':
        return _predict_multilabel_v3(text)
    if settings.model_variant == 'distilbert':
        transformer = _load_transformer()
        if transformer is None:
            return {'category': 'Model not installed', 'confidence': None,
                    'method': 'distilbert_artifact_missing'}
        import torch
        tokenizer, neural_model = transformer
        batch = tokenizer(text, truncation=True, max_length=192, return_tensors='pt')
        with torch.inference_mode():
            probs = torch.softmax(neural_model(**batch).logits, dim=-1)[0]
        index = int(probs.argmax().item())
        return {'category': neural_model.config.id2label[index],
                'confidence': round(float(probs[index].item()), 4),
                'method': 'finetuned_distilbert'}
    model = _load_baseline()
    if model is not None:
        category = str(model.predict([text])[0])
        confidence = float(max(model.predict_proba([text])[0])) if hasattr(model, 'predict_proba') else None
        return {'category': category, 'confidence': confidence, 'method': 'trained_tfidf_logreg'}

    text = text.lower()
    scores = {k: sum(token in text for token in tokens) for k, tokens in CATEGORY_KEYWORDS.items()}
    category = max(scores, key=scores.get)
    return {
        'category': category if scores[category] else 'Uncategorized',
        'confidence': None,
        'method': 'keyword_demo_only',
    }
