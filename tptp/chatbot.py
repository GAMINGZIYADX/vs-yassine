"""Chatbot d'orientation : classification d'intention + correction des fautes de frappe.

Lancer :  python chatbot.py
Dependances : spacy (+ fr_core_news_sm), pyspellchecker, tensorflow/keras, numpy
"""
import json
import os
import random
import re
import string

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # masquer les logs TensorFlow

import numpy as np
import spacy
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
from spellchecker import SpellChecker

DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_JSON = os.path.join(DOSSIER, "intents_chatbot_orientation.json")
FICHIER_MODELE = os.path.join(DOSSIER, "mychatbotmodel_v001.keras")
SEUIL_CONFIANCE = 0.60  # en dessous : on demande de reformuler
EPOCHS = 10

# ---------------------------------------------------------------------------
# 1. Chargement des donnees
# ---------------------------------------------------------------------------
with open(FICHIER_JSON, "r", encoding="UTF-8") as f:
    data = json.load(f)

input_texts, output_intentions = [], []
classes = [intent["tag"] for intent in data["intents"]]
reponses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
for intent in data["intents"]:
    for question in intent["patterns"]:
        input_texts.append(question)
        output_intentions.append(intent["tag"])

# ---------------------------------------------------------------------------
# 2. Tokenization + lemmatisation (spaCy) -> vocabulaire
# ---------------------------------------------------------------------------
nlp = spacy.load("fr_core_news_sm")

def lemmes(texte):
    return [t.lemma_ for t in nlp(texte.lower()) if t.lemma_ not in string.punctuation]

list_tockens = []
for phrase in input_texts:
    list_tockens.extend(lemmes(phrase))
list_tockens = list(dict.fromkeys(list_tockens))  # valeurs uniques, ordre conserve
index_tockens = {tok: i for i, tok in enumerate(list_tockens)}

def vectoriser(texte):
    vect = [0] * len(list_tockens)
    for lem in lemmes(texte):
        if lem in index_tockens:
            vect[index_tockens[lem]] = 1
    return vect

# ---------------------------------------------------------------------------
# 3. Correction orthographique (sans modifier le corpus)
# ---------------------------------------------------------------------------
spell = SpellChecker(language="fr")

# Mots du corpus (ex: salam, pole, python) : proteges et prioritaires
mots_corpus = set()
for phrase in input_texts:
    mots_corpus.update(re.findall(r"\w+", phrase.lower()))
spell.word_frequency.load_words(mots_corpus)

def corriger_mot(m):
    if not m.isalpha() or m in spell:
        return m
    candidats = spell.candidates(m) or set()
    dans_corpus = candidats & mots_corpus
    if dans_corpus:
        return max(dans_corpus, key=spell.word_usage_frequency)
    return spell.correction(m) or m

def corriger(texte):
    # ne remplace que les mots fautifs : ponctuation et espaces restent intacts
    return re.sub(r"[^\W\d_]+", lambda mo: corriger_mot(mo.group(0)), texte.lower())

# ---------------------------------------------------------------------------
# 4. Donnees d'entrainement + modele
# ---------------------------------------------------------------------------
def entrainer():
    training = []
    for phrase, intent in zip(input_texts, output_intentions):
        vect_y = [0] * len(classes)
        vect_y[classes.index(intent)] = 1
        training.append([vectoriser(phrase), vect_y])
    random.shuffle(training)
    train_X = np.array([t[0] for t in training])
    train_y = np.array([t[1] for t in training])

    model = Sequential([
        Input(shape=(len(list_tockens),)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(len(classes), activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=["accuracy"])
    model.fit(train_X, train_y, epochs=EPOCHS, verbose=1)
    model.save(FICHIER_MODELE)
    return model

def charger_ou_entrainer():
    if os.path.exists(FICHIER_MODELE):
        model = tf.keras.models.load_model(FICHIER_MODELE)
        # si le corpus a change (taille du vocabulaire/classes), on re-entraine
        if model.input_shape[-1] == len(list_tockens) and model.output_shape[-1] == len(classes):
            return model
    return entrainer()

# ---------------------------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------------------------
def predire(model, texte, afficher_scores=False):
    corrige = corriger(texte)
    proba = model.predict(np.array([vectoriser(corrige)]), verbose=0)[0]
    ordre = np.argsort(proba)[::-1]
    if afficher_scores:
        print(f"  texte corrigé : {corrige}")
        for i in ordre[:3]:
            print(f"  {classes[i]:<20} {proba[i] * 100:5.1f} %")
    return classes[ordre[0]], float(proba[ordre[0]])

def repondre(intention, confiance):
    if confiance < SEUIL_CONFIANCE:
        return "Je n'ai pas bien compris. Pouvez-vous reformuler ?"
    return random.choice(reponses[intention])

# ---------------------------------------------------------------------------
# 6. Boucle de conversation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--retrain" in sys.argv and os.path.exists(FICHIER_MODELE):
        os.remove(FICHIER_MODELE)
    debug = "--debug" in sys.argv

    model = charger_ou_entrainer()
    print("\nChatbot prêt ! (tapez 'quit' pour sortir)\n")
    while True:
        try:
            texte = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not texte:
            continue
        if texte.lower() in {"quit", "exit", "q"}:
            break
        intention, confiance = predire(model, texte, afficher_scores=debug)
        print(f"Bot : {repondre(intention, confiance)}")
        if intention == "au_revoir" and confiance >= SEUIL_CONFIANCE:
            break
