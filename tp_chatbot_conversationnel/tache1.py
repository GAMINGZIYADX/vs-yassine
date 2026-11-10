# importation
import os
os.environ["KERAS_BACKEND"] = "torch"  # Keras seul, sans TensorFlow (doit précéder l'import de keras)

import json
import random
import string
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam


nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

# 1. chargement du fichier json
json_path = Path(__file__).resolve().parent / "ia_chat_data (2).json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. affichage de toutes les intentions
for intent in data["intents"]:
    print(intent["intent"])

# 3. création des listes
classes = []            # a. intentions uniques
input_texts = []        # b. questions / patterns
output_intentions = []  # c. intention de chaque question
tokens = []             # d. tokens lemmatisés

lemmatizer = WordNetLemmatizer()
 
for intent in data["intents"]:
    if intent["intent"] not in classes:
        classes.append(intent["intent"])
 
    for question in intent["questions"]:
        input_texts.append(question)
        output_intentions.append(intent["intent"])
 
        # i. tokenization par mot
        for tok in nltk.word_tokenize(question):
            # ii. supprimer les tokens qui ne contiennent que des symboles
            if all(c in string.punctuation for c in tok):
                continue
            # iii. minuscules puis lemmatisation
            tokens.append(lemmatizer.lemmatize(tok.lower()))
 
print("\nclasses :", classes)
print("\ninput_texts :", input_texts)
print("\noutput_intentions :", output_intentions)
print("\ntokens :", tokens)
#tache II
vocab = sorted(set(tokens))
#creation
#Une List training qui contient les représentations numériques de chaque question/pattern X et les représentations numérique des intentions correspondantes Target
trainning = []
for text, intent_name in zip(input_texts, output_intentions):
    # mots (lemmatisés) de cette question
    words = []
    for tok in nltk.word_tokenize(text):
        if all(c in string.punctuation for c in tok):
            continue
        words.append(lemmatizer.lemmatize(tok.lower()))

    # i. One-Hot encoding de la question (sac de mots sur le vocabulaire)
    X = [1 if w in words else 0 for w in vocab]
    # One-Hot encoding de l'intention (Target)
    target = [0] * len(classes)
    target[classes.index(intent_name)] = 1

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

# 2. modèle Deep Learning
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
model.summary()

# 3. entraînement du modèle
history = model.fit(
    train_x, train_y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
)

# tache IV : prédiction
def bag_of_words(text):
    words = [
        lemmatizer.lemmatize(tok.lower())
        for tok in nltk.word_tokenize(text)
        if not all(c in string.punctuation for c in tok)
    ]
    return np.array([[1 if w in words else 0 for w in vocab]], dtype=np.float32)


SEUIL = 0.7  # score minimal pour accepter une prédiction


def predict_intent(text):
    bow = bag_of_words(text)
    # aucun mot connu du vocabulaire : le modèle n'a rien à analyser
    if bow.sum() == 0:
        return None, 0.0
    probs = model.predict(bow, verbose=0)[0]
    idx = int(np.argmax(probs))
    if probs[idx] < SEUIL:
        return None, float(probs[idx])
    return classes[idx], float(probs[idx])


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
    intention, score = predict_intent(question)
    if bag_of_words(question).sum() > 0:
        probs = model.predict(bag_of_words(question), verbose=0)[0]
        for i in np.argsort(probs)[::-1]:  # du score le plus élevé au plus bas
            print(f"  Classe {i} | {classes[i]:<30} | {probs[i]:.2%}")
    if intention is None:
        print(f"Intention : inconnue (je n'ai pas compris) | Score : {score:.2%}")
        continue
    print(f"Intention : {intention} | Classe : {classes.index(intention)} | Score : {score:.2%}")
