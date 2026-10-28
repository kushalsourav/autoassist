"""V3 inference tests do not need the 268MB trained weights."""
from types import SimpleNamespace
import torch
import backend.app.classifier as classifier
from backend.app.schemas import Classification


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs['max_length'] == 128
        return {'input_ids': torch.tensor([[1, 2]])}


class DummyModel:
    config = SimpleNamespace(id2label={0: 'Electrical', 1: 'Brakes', 2: 'Engine / Fuel'})

    def __call__(self, **kwargs):
        # sigmoid(-0.5)=0.378, sigmoid(1.4)=0.802, sigmoid(-3)=0.047
        return SimpleNamespace(logits=torch.tensor([[-0.5, 1.4, -3.0]]))


def test_multilabel_sigmoid_returns_multiple_groups(monkeypatch):
    monkeypatch.setattr(classifier, '_load_multilabel_v3',
                        lambda: (DummyTokenizer(), DummyModel(), 0.35, 128))
    result = classifier._predict_multilabel_v3('Brakes and wiring faults')
    assert result['category'] == 'Brakes'
    assert [label['category'] for label in result['labels']] == ['Brakes', 'Electrical']
    assert result['method'] == 'finetuned_multilabel_distilbert_v3'
    Classification(**result)  # Must remain compatible with FastAPI response model.


def test_no_label_above_threshold_is_uncertain(monkeypatch):
    monkeypatch.setattr(classifier, '_load_multilabel_v3',
                        lambda: (DummyTokenizer(), DummyModel(), 0.99, 128))
    result = classifier._predict_multilabel_v3('Unclear issue')
    assert result['category'] == 'Uncertain'
    assert result['labels'] == []
    assert result['confidence'] is None


def test_missing_model_does_not_fall_back_to_keyword_demo(monkeypatch):
    monkeypatch.setattr(classifier, '_load_multilabel_v3', lambda: None)
    result = classifier._predict_multilabel_v3('Engine stalls')
    assert result['method'] == 'multilabel_v3_artifact_missing'
    assert result['confidence'] is None
