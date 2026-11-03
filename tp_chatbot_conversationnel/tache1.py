# importation
import json
import string
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer


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