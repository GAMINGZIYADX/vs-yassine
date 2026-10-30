import random
import re
from nltk.util import ngrams

# 1. Corpus
corpus = """Le chat mange une souris.
Le chien aboie dans la cour.
Le chat dort sur le canapé.
Le chien mange une croquette.
Le chat joue avec une pelote de laine.
Le chien joue avec une balle.
Le chat saute sur la table.
Le chien court après la voiture.
La souris se cache sous le canapé.
Le chat regarde par la fenêtre.
La souris mange du fromage.
Le chien dort dans la cuisine.
La voiture passe devant la maison.
Le chat boit du lait.
La souris court dans la cuisine.
Le chien regarde la souris.
Le chat et le chien jouent dans la cour.
Une croquette tombe sous la table.
La maison a une grande fenêtre.
Le chat attend la souris près du canapé.
L'oiseau chante sur le toit de la maison.
L'oiseau mange une graine dans la cour.
Le chat regarde l'oiseau par la fenêtre.
Le chien aboie après l'oiseau.
L'enfant joue avec le chien dans la cour.
L'enfant donne une croquette au chien.
L'enfant donne du lait au chat.
La maman prépare le repas dans la cuisine.
Le papa lit le journal sur le canapé.
Le chat dort sur les genoux du papa.
Le chien dort près de la table.
La maman donne du fromage à l'enfant.
L'enfant mange du fromage dans la cuisine.
Le papa regarde la voiture devant la maison.
La voiture roule dans la rue.
Le chien court dans la rue après la balle.
Le chat court après la souris dans la cuisine.
La souris mange une graine sous la table.
La souris regarde le chat par le trou.
Le chat saute sur le canapé et dort.
Le chien saute sur la table et mange le fromage.
L'enfant rit quand le chien joue avec la balle.
La maman rit quand le chat saute sur la table.
Le papa boit du café dans la cuisine.
La maman boit du lait avec l'enfant.
Le soleil brille sur la maison.
Le soleil entre par la fenêtre de la cuisine.
Le chat dort au soleil près de la fenêtre.
Le chien dort au soleil dans la cour.
La pluie tombe sur le toit de la maison.
Le chat regarde la pluie par la fenêtre.
Le chien rentre dans la maison sous la pluie.
L'enfant joue avec la pelote de laine.
Le chat joue avec la pelote de laine sur le canapé.
La balle roule sous la table.
Le chien cherche la balle sous le canapé.
Le chat cherche la souris dans la cuisine.
L'oiseau regarde le chat sur le toit.
La nuit tombe sur la maison.
Le chat et la souris dorment dans la maison.
Le chien garde la maison pendant la nuit.
Le matin, le soleil brille sur la cour.
Le matin, la maman prépare le café dans la cuisine.
Le matin, le chat boit du lait et le chien mange une croquette.
"""

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