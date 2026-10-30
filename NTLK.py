#1.importation des library nltk
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")
from collections import defaultdict
from collections import Counter
import random
#2.Corpus
corpus = """
    Le chat mange une souris.
    Le chien aboie dans la cour.
    Le chat dort sur le cannape.
    Le chien mange une croquette.
    Le chat joue avec une pelote de laine.
    Le chien joue avec une balle.
    Le chat saute sur la table.
    Le chien court apres la voiture.
    La souris se cache sous le canape.
    Le chat regarde par la fenetre.
    """

tokens = word_tokenize(corpus.lower(), language="french")
print("Tokens :", tokens)
# 3.Creation des trigrammes avec ngram
N = 3
liste_ngrams = list(ngrams(tokens,N))
print("\nQuelque trigrammes :")
for ng in liste_ngrams:
    print(ng)
