import random
import re
from nltk.util import ngrams

# 1. Corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

FIN = {".", "!", "?"}
N = 3


# 2. Tokenisation (regex : pas besoin de télécharger punkt)
#    "l'oiseau" -> "l'", "oiseau" ; le point devient un token à part
def tokeniser(texte):
    return re.findall(r"\w+'?|[.!?]", texte.lower())


tokens = tokeniser(corpus)

# 3. Trigrammes -> {(w1, w2): {w3: compte}}
dictionnaire = {}
for w1, w2, w3 in ngrams(tokens, N):
    # on ignore les trigrammes qui traversent une frontière de phrase
    if w1 in FIN or w2 in FIN:
        continue
    dictionnaire.setdefault((w1, w2), {})
    dictionnaire[(w1, w2)][w3] = dictionnaire[(w1, w2)].get(w3, 0) + 1

# 4. Comptes -> pourcentages
for cle, suivants in dictionnaire.items():
    total = sum(suivants.values())
    for mot in suivants:
        suivants[mot] = suivants[mot] / total * 100

# Décommente pour voir le modèle :
# for cle, suivants in dictionnaire.items():
#     print(cle, ":", suivants)


# 5. Génération
def generer_une_phrase(deux_mots, max_len=25):
    phrase = list(deux_mots)
    while len(phrase) < max_len:
        cle = tuple(phrase[-2:])
        if cle not in dictionnaire:
            return None  # impasse
        mots = list(dictionnaire[cle].keys())
        poids = list(dictionnaire[cle].values())
        mot = random.choices(mots, weights=poids)[0]
        phrase.append(mot)
        if mot in FIN:
            return phrase  # fin propre
    return None  # trop long, probablement une boucle


def mise_en_forme(phrase):
    texte = " ".join(phrase)
    texte = re.sub(r"\s+([.!?])", r"\1", texte)  # "chat ." -> "chat."
    texte = texte.replace("' ", "'")             # "l' enfant" -> "l'enfant"
    return texte.capitalize()


def generer(essais=200):
    saisie = input("\nEntrer deux mots (q pour quitter) : ").strip().lower()
    if saisie == "q":
        return False

    deux_mots = tokeniser(saisie)
    if len(deux_mots) != 2:
        print("Il faut exactement deux mots.")
        return True
    if tuple(deux_mots) not in dictionnaire:
        print("Ces deux mots ne sont pas dans le corpus.")
        return True

    for _ in range(essais):
        phrase = generer_une_phrase(deux_mots)
        if phrase:
            print("Phrase générée :", mise_en_forme(phrase))
            return True
    print("Impossible de générer une phrase complète.")
    return True


if __name__ == "__main__":
    while generer():
        pass