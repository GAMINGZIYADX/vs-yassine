"""Reconnaissance d'entités avec DistilCamemBERT-NER fine-tuné sur nos données.

Le modèle est entraîné par ner_train.py et sauvegardé dans le dossier ner_model/.
"""
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "ner_model"

# un mot = lettres/chiffres (avec tirets internes) ou un symbole isolé : "l'IA" -> l ' IA, "500 €" -> 500 €
WORD_RE = re.compile(r"\w+(?:-\w+)*|[^\w\s]")


def tokenize(text):
    """Retourne [(mot, début, fin), ...]."""
    text = text.replace("’", "'")
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


_tokenizer = None
_model = None


def model_available():
    return (MODEL_DIR / "config.json").exists()


def _load():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def predict_entities(text, min_score=0.5):
    """Retourne [(texte, type, début, fin), ...] ; [] si le modèle n'est pas entraîné."""
    if not model_available():
        return []
    import torch

    tokenizer, model = _load()
    text = text.replace("’", "'")
    toks = tokenize(text)
    if not toks:
        return []
    enc = tokenizer([t[0] for t in toks], is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits[0], dim=-1)
    word_ids = enc.word_ids()
    id2label = model.config.id2label

    # étiquette du premier sous-mot de chaque mot
    labels = ["O"] * len(toks)
    scores = [1.0] * len(toks)
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        idx = int(probs[pos].argmax())
        labels[wid] = id2label[idx]
        scores[wid] = float(probs[pos][idx])

    # regroupe les B-/I- en entités
    entities, cur = [], None
    for (word, s, e), lab, sc in zip(toks, labels, scores):
        if lab == "O" or sc < min_score:
            if cur:
                entities.append(cur)
                cur = None
            continue
        prefix, etype = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[1] != etype:
            if cur:
                entities.append(cur)
            cur = [s, etype, e]
        else:
            cur[2] = e
    if cur:
        entities.append(cur)
    return [(text[s:e], etype, s, e) for s, etype, e in entities]
