# importation
import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras seul, sans TensorFlow (doit précéder l'import de keras)

import json
import random
import re
import string
import sys
from functools import lru_cache
from pathlib import Path

import nltk
import numpy as np
import keras
from nltk.stem.snowball import FrenchStemmer
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam

# la console Windows n'est pas toujours en UTF-8 : évite les erreurs d'affichage des accents
sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
RETRAIN = "--retrain" in sys.argv  # python tache1.py --retrain : ignore le modèle sauvegardé

# ne télécharge que si les données manquent (évite un appel réseau à chaque lancement)
for _res, _pkg in (("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")):
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# 1. chargement du fichier json
json_path = BASE_DIR / "ia_chat_data (2).json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. affichage de toutes les intentions
for intent in data["intents"]:
    print(intent["intent"])

# 3. création des listes
classes = []            # a. intentions uniques
input_texts = []        # b. questions / patterns
output_intentions = []  # c. intention de chaque question
tokens = []             # d. tokens racinisés
responses = {}          # e. réponses possibles par intention

# Le corpus est en français : tokenizer français + stemmer français.
# (WordNetLemmatizer est anglais seulement, il ne changeait quasiment rien ici.)
stemmer = FrenchStemmer()


def normaliser(text):
    """l'IA -> l IA : sépare les élisions pour que 'IA' soit reconnu seul."""
    return text.replace("’", "'").replace("'", " ")


@lru_cache(maxsize=None)  # chaque question d'entraînement n'est tokenisée qu'une fois
def tokenize(text):
    """Tokens d'une phrase : sans symboles seuls, en minuscules, racinisés."""
    return tuple(
        stemmer.stem(tok.lower())
        for tok in nltk.word_tokenize(normaliser(text), language="french")
        if not all(c in string.punctuation for c in tok)
    )


for intent in data["intents"]:
    if intent["intent"] not in classes:
        classes.append(intent["intent"])
    responses.setdefault(intent["intent"], []).extend(intent.get("responses", []))

    for question in intent["questions"]:
        input_texts.append(question)
        output_intentions.append(intent["intent"])
        tokens.extend(tokenize(question))

print("\nclasses :", classes)
print("\ninput_texts :", input_texts)
print("\noutput_intentions :", output_intentions)
print("\ntokens :", tokens)

# tache II
vocab = sorted(set(tokens))
vocab_index = {w: i for i, w in enumerate(vocab)}      # mot -> position dans le vecteur
class_index = {c: i for i, c in enumerate(classes)}    # intention -> position dans la cible
# Une liste training qui contient la représentation numérique de chaque question X
# et la représentation numérique de l'intention correspondante (Target)


def bag_of_words(text):
    bow = np.zeros((1, len(vocab)), dtype=np.float32)
    for w in tokenize(text):
        i = vocab_index.get(w)
        if i is not None:  # les mots hors vocabulaire sont ignorés
            bow[0, i] = 1
    return bow


trainning = []
for text, intent_name in zip(input_texts, output_intentions):
    # i. One-Hot encoding de la question (sac de mots sur le vocabulaire)
    X = bag_of_words(text)[0].tolist()
    # One-Hot encoding de l'intention (Target)
    target = [0] * len(classes)
    target[class_index[intent_name]] = 1

    trainning.append([X, target])

# b. mélanger les éléments de la liste
random.shuffle(trainning)

# c. convertir la liste en array
trainning = np.array(trainning, dtype=object)

# d. séparer train_x et train_y
train_x = np.array(list(trainning[:, 0]), dtype=np.float32)
train_y = np.array(list(trainning[:, 1]), dtype=np.float32)

print("\nvocab size :", len(vocab))
print("train_x shape :", train_x.shape)
print("train_y shape :", train_y.shape)

# tache III : réseau de neurones avec Keras
# 1. définition de quelques paramètres
input_shape = (len(train_x[0]),)   # taille du vocabulaire
output_shape = len(train_y[0])     # nombre d'intentions
epochs = 100
batch_size = 8
VAL_RATIO = 0.2                    # part des questions gardée pour évaluer le modèle


# 2. modèle Deep Learning
def build_model():
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(output_shape, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def split_train_val(y, ratio):
    """Découpe stratifiée : ~ratio de chaque intention (au moins 1) va en validation."""
    labels = y.argmax(axis=1)
    val_idx = []
    for c in range(y.shape[1]):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        val_idx.extend(idx[: max(1, round(len(idx) * ratio))])
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True
    return ~val_mask, val_mask


MODEL_PATH = BASE_DIR / "chatbot_model.keras"
META_PATH = BASE_DIR / "chatbot_meta.json"


def load_saved_model():
    """Recharge le modèle sauvegardé s'il correspond au vocabulaire et aux intentions actuels."""
    if RETRAIN or not (MODEL_PATH.exists() and META_PATH.exists()):
        return None, None
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if meta["vocab"] != vocab or meta["classes"] != classes:
            return None, None  # le JSON a changé : il faut réentraîner
        return load_model(MODEL_PATH), meta
    except Exception:
        return None, None


model, meta = load_saved_model()
if model is not None:
    print(f"\nModèle rechargé depuis {MODEL_PATH.name} "
          f"(accuracy de validation lors de l'entraînement : {meta['val_accuracy']:.2%})")
    print("(utilisez --retrain pour réentraîner)")
else:
    # 3a. entraînement sur un sous-ensemble, évaluation sur les questions jamais vues
    tr, va = split_train_val(train_y, VAL_RATIO)
    eval_model = build_model()
    eval_model.summary()
    eval_model.fit(train_x[tr], train_y[tr], epochs=epochs, batch_size=batch_size, verbose=0)
    val_loss, val_acc = eval_model.evaluate(train_x[va], train_y[va], verbose=0)
    print(f"\nValidation ({va.sum()} questions non vues) : accuracy = {val_acc:.2%} | loss = {val_loss:.3f}")

    # 3b. entraînement final sur toutes les questions
    model = build_model()
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(MODEL_PATH)
    META_PATH.write_text(
        json.dumps({"vocab": vocab, "classes": classes, "val_accuracy": float(val_acc)}, ensure_ascii=False),
        encoding="utf-8",
    )

# tache IV : prédiction
SEUIL = 0.7  # score minimal pour accepter une prédiction


def predict_intent(text):
    """Retourne (intention ou None, score, probabilités ou None). Un seul appel au modèle."""
    bow = bag_of_words(text)
    # aucun mot connu du vocabulaire : le modèle n'a rien à analyser
    if bow.sum() == 0:
        return None, 0.0, None
    # appel direct du modèle : bien plus rapide que model.predict pour un seul exemple
    probs = keras.ops.convert_to_numpy(model(bow, training=False))[0]
    idx = int(np.argmax(probs))
    if probs[idx] < SEUIL:
        return None, float(probs[idx]), probs
    return classes[idx], float(probs[idx]), probs


FALLBACK = [
    "Désolé, je n'ai pas compris. Pouvez-vous reformuler ?",
    "Hmm, je ne suis pas sûr de comprendre. Pouvez-vous préciser votre question ?",
]


def reply(intention):
    """Réponse aléatoire parmi celles de l'intention (message d'excuse si intention inconnue)."""
    options = responses.get(intention) if intention else None
    return random.choice(options or FALLBACK)


# tache V : détection des entités (dictionnaire "entities" du JSON)
# {"entities": {"type": ["valeur1", "valeur2", ...]}} : à enrichir avec le corpus
def word_regex(value):
    """Regex compilée qui reconnaît `value` comme mot entier, sans tenir compte de la casse."""
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)", flags=re.IGNORECASE)


entity_values = [
    (value, etype)
    for etype, values in data.get("entities", {}).items()
    for value in values
]
entity_values.sort(key=lambda ev: len(ev[0]), reverse=True)  # les plus longues d'abord
entity_patterns = [(word_regex(value), etype) for value, etype in entity_values]  # compilées une seule fois
PRIX_RE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:€|euros?|eur|dhs?|dirhams?|\$)(?!\w)", flags=re.IGNORECASE)


def detect_entities(text):
    text = text.replace("’", "'")
    found = []
    taken = []  # zones déjà prises, pour éviter les chevauchements
    for pattern, etype in entity_patterns:
        for m in pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.group(0), etype))
    # montants chiffrés : "500 €", "2000 euros", "300 dh"
    for m in PRIX_RE.finditer(text):
        if not any(m.start() < e and m.end() > s for s, e in taken):
            taken.append((m.start(), m.end()))
            found.append((m.group(0), "prix"))
    return found


# type de question (comment, pourquoi, qu'est-ce que...) : mots-clés du JSON "question_types"
question_patterns = [
    (word_regex(kw), kw, qtype)
    for qtype, kws in data.get("question_types", {}).items()
    for kw in kws
]


def detect_question_type(text):
    """Retourne le type dont le mot-clé apparaît en premier dans la phrase (None sinon)."""
    t = text.replace("’", "'")
    best = None  # (position, -longueur, type)
    for pattern, kw, qtype in question_patterns:
        m = pattern.search(t)
        if m:
            cand = (m.start(), -len(kw), qtype)
            if best is None or cand < best:
                best = cand
    return best[2] if best else None


# définitions courtes des langages / frameworks : utilisées quand on demande "c'est quoi X ?"
DEFINITIONS = {
    "python": "Python est un langage de programmation simple et polyvalent, le plus utilisé en IA grâce à son écosystème de bibliothèques (NumPy, Pandas, scikit-learn, PyTorch...).",
    "r": "R est un langage orienté statistiques et analyse de données, très utilisé en recherche et en visualisation.",
    "java": "Java est un langage généraliste robuste, utilisé surtout pour les applications d'entreprise et le Big Data.",
    "c++": "C++ est un langage compilé très rapide, utilisé quand la performance compte (le cœur de TensorFlow ou PyTorch est écrit en C++).",
    "sql": "SQL est le langage standard pour interroger et manipuler les bases de données, indispensable pour préparer les données.",
    "julia": "Julia est un langage récent conçu pour le calcul scientifique, rapide comme le C++ et simple comme Python.",
    "tensorflow": "TensorFlow est une bibliothèque de deep learning développée par Google, adaptée à la production et au déploiement.",
    "keras": "Keras est une API haut niveau pour construire des réseaux de neurones simplement (compatible TensorFlow, PyTorch ou JAX).",
    "pytorch": "PyTorch est une bibliothèque de deep learning développée par Meta, très appréciée en recherche pour sa flexibilité.",
    "scikit-learn": "scikit-learn est la bibliothèque Python de référence pour le machine learning classique (régression, classification, clustering).",
    "pandas": "Pandas est une bibliothèque Python pour manipuler et analyser des données tabulaires (DataFrames).",
    "numpy": "NumPy est la bibliothèque Python de calcul numérique sur tableaux et matrices, la base de presque tout l'écosystème IA.",
    "jax": "JAX est une bibliothèque de Google pour le calcul numérique haute performance et la différentiation automatique.",
}


def define(qtype, entities):
    """Définition de la première entité connue si la question est de type 'définition' (sinon None)."""
    if not qtype or not qtype.startswith("définition"):
        return None
    for value, _ in entities:
        if value.casefold() in DEFINITIONS:
            return DEFINITIONS[value.casefold()]
    return None


# tache VI : état de la conversation (State.md), mis à jour après chaque message
STATE_PATH = BASE_DIR / "State.md"
# intentions qui ne mettent pas l'état à jour (comparaison insensible à la casse)
IGNORER = {"salutation", "au revoir", "remerciement"}
SLOT_TTL = 3  # une entité est oubliée après ce nombre de messages sans être mentionnée à nouveau

# slots affichés dans State.md (dans cet ordre) : "intention", "type_question" puis les types d'entités
SLOTS = ["intention", "type_question"] + list(data.get("entities", {}).keys())
state = {slot: None for slot in SLOTS}
slot_age = {}  # type d'entité -> nombre de messages écoulés depuis sa dernière mention
historique = []

unknown_intents = IGNORER - {c.casefold() for c in classes}
if unknown_intents:
    print(f"\nAttention : intentions de IGNORER absentes du JSON : {sorted(unknown_intents)}")


def update_state(question, intention, qtype, entities):
    if intention is not None and intention.casefold() in IGNORER:
        return False
    # vieillissement : les entités non mentionnées depuis SLOT_TTL messages sont oubliées
    for etype in list(slot_age):
        slot_age[etype] += 1
        if slot_age[etype] > SLOT_TTL:
            state[etype] = None
            del slot_age[etype]
    if intention is not None:  # intention inconnue : on garde la précédente
        state["intention"] = intention
        if qtype:
            state["type_question"] = qtype
    for value, etype in entities:  # la valeur la plus récente remplace l'ancienne
        state[etype] = value
        slot_age[etype] = 0
    historique.append((question, intention or "inconnue"))
    return True


def write_state():
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        f.write("# État du chatbot\n\n")
        f.write("| Slot | Valeur |\n|---|---|\n")
        for slot in SLOTS:
            f.write(f"| {slot} | {state[slot] if state[slot] else '—'} |\n")
        f.write("\n## Historique\n")
        for i, (q, it) in enumerate(historique, 1):
            f.write(f"{i}. {q} → *{it}*\n")


write_state()  # crée State.md dès le démarrage (état vide)

print("\n--- Prédiction (tapez 'quit' pour arrêter) ---")
while True:
    try:
        question = input("Vous : ").strip()
    except EOFError:
        break
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue
    qtype = detect_question_type(question)
    if qtype:
        print("  Type de question :", qtype)
    entities = detect_entities(question)
    if entities:
        print("  Entités :", ", ".join(f"{v} ({t})" for v, t in entities))
    intention, score, probs = predict_intent(question)
    if probs is not None:
        for i in np.argsort(probs)[::-1]:  # du score le plus élevé au plus bas
            print(f"  Classe {i} | {classes[i]:<30} | {probs[i]:.2%}")
    if intention is None:
        print(f"Intention : inconnue (je n'ai pas compris) | Score : {score:.2%}")
    else:
        print(f"Intention : {intention} | Classe : {class_index[intention]} | Score : {score:.2%}")
    print("Bot :", define(qtype, entities) or reply(intention))
    if update_state(question, intention, qtype, entities):
        write_state()
