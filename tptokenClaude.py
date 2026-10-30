import random
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('punkt_tab')

# 1. Corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

# 2. Tokenisation
tokens = word_tokenize(corpus.lower(), language="french")

# 3. Trigrammes -> dictionnaire {(w1, w2): {w3: proba}}
N = 3
FIN = {".", "!", "?"}
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

for cle, suivants in dictionnaire.items():
    print(cle, ":", suivants)


# 5. Génération
def generer(max_len=20):
    deux_mots = input("\nEntrer deux mots : ").lower().split()
    if len(deux_mots) != 2:
        print("Il faut exactement deux mots.")
        return

    phrase = deux_mots.copy()
    while len(phrase) < max_len:
        cle = tuple(phrase[-2:])
        if cle not in dictionnaire:
            break

        mots = list(dictionnaire[cle].keys())
        poids = list(dictionnaire[cle].values())
        mot_suivant = random.choices(mots, weights=poids)[0]

        phrase.append(mot_suivant)
        if mot_suivant in FIN:
            print(mot_suivant)
            break

    print("Phrase générée :", " ".join(phrase))


if __name__ == "__main__":
    generer()