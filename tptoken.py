import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('punkt_tab')
# 1. Corpus sous forme de texte
corpus = """Le chat mange une souris.
Le chien aboie dans la cour.
Le chat dort sur le canapé.
Le chien mange une croquette.
Le chat joue avec une pelote de laine.
Le chien joue avec une balle.
Le chat saute sur la table.
Le chien court après la voiture.
La souris se cache sous le canapé.
Le chat regarde par la fenêtre."""

# 2. Transformer le texte en mots
tokens = word_tokenize(corpus.lower() , language='french')
print('tokens : ', tokens)
N = 3
list_grams = list(ngrams(tokens,N))
print("\nQuelques trigrams : ")
# for ng in list_grams:
#     print(ng)

dictionnaire = {}
for gram in list_grams:
    deux_mots = (gram[0], gram[1])
    mot_suivant = gram[2]

    if deux_mots not in dictionnaire:
        dictionnaire[deux_mots] = {}

    if mot_suivant not in dictionnaire[deux_mots]:
        dictionnaire[deux_mots][mot_suivant] = 1
    else:
        dictionnaire[deux_mots][mot_suivant] += 1

# pourcentage
for deux_mots in dictionnaire:
    total = sum(dictionnaire[deux_mots].values())

    for mot in dictionnaire[deux_mots]:
        dictionnaire[deux_mots][mot] = (
            dictionnaire[deux_mots][mot] / total
        ) * 100

for deux_mots, mots in dictionnaire.items():
    print(deux_mots, ":", mots)


# generation une phrase
import random

def detect():
    deux_mots = input("Entrer deux mots : ").lower().split()
    phrase = deux_mots.copy()

    while True:
        dernieres_mots = tuple(phrase[-2:])

        # Chercher les mots suivants possibles
        if dernieres_mots not in dictionnaire:
            break

        mots_possibles = list(dictionnaire[dernieres_mots].keys())

        # Choisir un mot au hasard
        mot_suivant = random.choice(mots_possibles)

        phrase.append(mot_suivant)

        # Vérifier si la phrase est terminée
        if mot_suivant in [".", "!", "?"]:
            break

    print("Phrase générée :", " ".join(phrase))


detect()