"""Fine-tuning de DistilCamemBERT-NER sur nos entités (métier, langage, formation, prix...).

  python ner_train.py eval    # entraîne avec 75% des valeurs, teste sur des valeurs jamais vues (rien n'est sauvegardé)
  python ner_train.py final   # entraîne avec toutes les valeurs et sauvegarde dans ner_model/

Les étiquettes (BIO) sont générées automatiquement à partir de la section "entities" du JSON :
  - les questions du corpus sont étiquetées avec les listes de mots-clés ;
  - des phrases supplémentaires sont générées à partir de gabarits pour varier les contextes.
"""
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ner import MODEL_DIR, tokenize

BASE = "cmarkea/distilcamembert-base-ner"
HERE = Path(__file__).resolve().parent
EPOCHS, BATCH, LR = 6, 16, 5e-5

data = json.load(open(HERE / "ia_chat_data (2).json", encoding="utf-8"))
ENT = data["entities"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENT for p in ("B", "I")]
AMOUNT_RE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:€|euros?|eur|dhs?|dirhams?|\$)(?!\w)", re.IGNORECASE)

TEMPLATES = [
    "Je veux apprendre {langage} pour devenir {metier}",
    "Quel est le salaire d'un {metier} en {secteur} ?",
    "Quelle formation {modalite} choisir pour me former en {domaine_ia} ?",
    "Peut-on utiliser {framework} avec {langage} ?",
    "Y a-t-il un {formation} {modalite} en {domaine_ia} ?",
    "Est-ce que {plateforme} est {prix} ?",
    "Combien coûte un {formation} {modalite} ? Environ {montant}",
    "Je cherche un {metier} dans le secteur {secteur}",
    "Comment apprendre {framework} sur {plateforme} ?",
    "Je voudrais m'inscrire à un {formation} en {domaine_ia}",
    "La formation coûte {montant}, est-ce normal ?",
    "Je préfère étudier {modalite} plutôt que {modalite}",
    "Quels sont les débouchés en {secteur} avec un {formation} ?",
    "{langage} ou {langage} pour {domaine_ia} ?",
    "Existe-t-il un {formation} en {domaine_ia} avec {prix} ?",
    "Qu'est-ce que {framework} ?",
    "À quoi sert {domaine_ia} dans le secteur {secteur} ?",
    "Je suis {metier} et je veux me former à {framework}",
    "Où suivre des cours de {domaine_ia} {modalite} ?",
    "Quel budget prévoir, {montant} suffit-il pour un {formation} ?",
    "Peut-on travailler comme {metier} sans {formation} ?",
    "Avez-vous une offre {prix} pour {domaine_ia} ?",
    "Je veux suivre un cours {modalite} sur {plateforme}",
    "Quelle différence entre {framework} et {framework} ?",
    "Salut, je veux apprendre {domaine_ia} en {modalite}",
    "Mon budget est de {montant} par mois",
    "Quel {formation} pour devenir {metier} ?",
    "Je connais {langage} et {langage}, que me conseillez-vous ?",
    "Les {metier} sont-ils bien payés en {secteur} ?",
    "Combien de temps dure un {formation} {modalite} ?",
    "Je veux passer la certification {certification} à {etablissement}",
    "Quelles formations {formation} propose {etablissement} en {secteur} ?",
    "Je veux une certification {logiciel} en {modalite}",
    "Qu'est-ce que {certification} ?",
    "Comment passer {certification} chez {etablissement} ?",
    "Je maîtrise {logiciel} et {logiciel}, quelle {formation} choisir ?",
    "Combien coûte {certification} ?",
    "Y a-t-il une formation en {secteur} avec {certification} ?",
]


def random_amount():
    n = random.choice([30, 50, 80, 100, 200, 300, 450, 500, 800, 1000, 1500, 2000, 3000, 5000])
    return f"{n} {random.choice(['€', 'euros', 'EUR', 'dh'])}"


def find_spans(text, values):
    """Spans (début, fin, type) des valeurs de `values` {type: [valeurs]} trouvées dans text."""
    spans = []
    items = sorted(((v, t) for t, vs in values.items() for v in vs), key=lambda x: -len(x[0]))
    for v, t in items:
        for m in re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text, flags=re.IGNORECASE):
            if not any(m.start() < e and m.end() > s for s, e, _ in spans):
                spans.append((m.start(), m.end(), t))
    for m in AMOUNT_RE.finditer(text):
        if not any(m.start() < e and m.end() > s for s, e, _ in spans):
            spans.append((m.start(), m.end(), "prix"))
    return spans


def to_bio(text, spans):
    """(mots, étiquettes BIO) à partir des spans en caractères."""
    toks = tokenize(text)
    labels = ["O"] * len(toks)
    for ss, se, t in spans:
        first = True
        for i, (_, s, e) in enumerate(toks):
            if s >= ss and e <= se:
                labels[i] = ("B-" if first else "I-") + t
                first = False
    return [w for w, _, _ in toks], labels


def fill(template, values):
    out, spans, pos = "", [], 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out += template[pos:m.start()]
        t = m.group(1)
        v = random_amount() if t == "montant" else random.choice(values[t])
        spans.append((len(out), len(out) + len(v), "prix" if t == "montant" else t))
        out += v
        pos = m.end()
    return out + template[pos:], spans


def augment(text):
    if random.random() < 0.25:
        text = text.lower()
    if random.random() < 0.2:
        text = text.rstrip(" ?")
    return text


def generate(values, per_template):
    sents = []
    for tpl in TEMPLATES:
        for _ in range(per_template):
            text, spans = fill(tpl, values)
            sents.append(to_bio(augment(text), spans))  # minuscules / sans "?" : mêmes positions de spans
    return sents


def corpus_sentences(skip_values=()):
    skip = {v.lower() for v in skip_values}
    sents = []
    for it in data["intents"]:
        for q in it["questions"]:
            spans = find_spans(q.replace("’", "'"), ENT)
            if any(q[s:e].lower() in skip for s, e, _ in spans):
                continue  # contient une valeur réservée au test
            sents.append(to_bio(q, spans))
    return sents


def build_split(mode):
    random.seed(0)
    if mode == "final":
        return corpus_sentences() + generate(ENT, per_template=14), []
    train_vals, test_vals = {}, {}
    for t, vs in ENT.items():
        vs = vs[:]
        random.shuffle(vs)
        k = max(1, math.ceil(0.25 * len(vs)))
        test_vals[t], train_vals[t] = vs[:k], vs[k:]
    held = [v for vs in test_vals.values() for v in vs]
    train = corpus_sentences(skip_values=held) + generate(train_vals, per_template=14)
    test = generate(test_vals, per_template=5)
    print("valeurs réservées au test :", {t: v for t, v in test_vals.items()})
    return train, test, train_vals


def batches(sents, tokenizer, shuffle):
    idx = list(range(len(sents)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), BATCH):
        chunk = [sents[j] for j in idx[i:i + BATCH]]
        enc = tokenizer([w for w, _ in chunk], is_split_into_words=True, padding=True,
                        truncation=True, max_length=96, return_tensors="pt")
        lab = torch.full(enc["input_ids"].shape, -100)
        for b, (_, labels) in enumerate(chunk):
            prev = None
            for p, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid != prev:
                    lab[b, p] = LABELS.index(labels[wid])
                prev = wid
        yield enc, lab, chunk


def predict(model, tokenizer, sents):
    model.eval()
    preds = []
    with torch.no_grad():
        for enc, _, chunk in batches(sents, tokenizer, shuffle=False):
            out = model(**enc).logits.argmax(-1)
            for b, (words, _) in enumerate(chunk):
                labels, prev = ["O"] * len(words), None
                for p, wid in enumerate(enc.word_ids(b)):
                    if wid is not None and wid != prev:
                        labels[wid] = LABELS[int(out[b, p])]
                    prev = wid
                preds.append(labels)
    return preds


def main(mode):
    split = build_split(mode)
    train, test = split[0], split[1]
    print(f"{len(train)} phrases d'entraînement, {len(test)} de test")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, ignore_mismatched_sizes=True,
    )
    # force nos étiquettes (le modèle de base avait 5 classes, avertissement de transformers)
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {l: i for i, l in enumerate(LABELS)}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * math.ceil(len(train) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for enc, lab, _ in batches(train, tokenizer, shuffle=True):
            loss = model(**enc, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        print(f"époque {epoch}/{EPOCHS} - loss moyenne : {total / math.ceil(len(train) / BATCH):.4f}")

    if mode == "eval":
        gold = [l for _, l in test]
        pred = predict(model, tokenizer, test)
        print("\n=== Modèle sur des valeurs JAMAIS vues à l'entraînement ===")
        print(classification_report(gold, pred, digits=3))
        # référence : la recherche par mots-clés avec seulement les valeurs d'entraînement
        # (les mots sont remis en texte, les spans sont recalculés puis réétiquetés)
        base_pred = [to_bio(" ".join(w), find_spans(" ".join(w), split[2]))[1] for w, _ in test]
        print(f"F1 modèle : {f1_score(gold, pred):.3f} | F1 mots-clés seuls : {f1_score(gold, base_pred):.3f}")
    else:
        MODEL_DIR.mkdir(exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print("modèle sauvegardé dans", MODEL_DIR)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval")
