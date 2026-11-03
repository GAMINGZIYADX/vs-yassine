"""
Word embeddings from scratch on the small corpus:
co-occurrence -> PPMI -> SVD -> cosine similarity
+ a bigram model for next-word prediction.
pip install numpy
"""
import re
from collections import Counter, defaultdict
import numpy as np

corpus = """Le chat mange une souris. Le chien aboie dans la cour. Le chat dort sur le canapé. Le chien mange une croquette. Le chat joue avec une pelote de laine. Le chien joue avec une balle. Le chat saute sur la table. Le chien court après la voiture. La souris se cache sous le canapé. Le chat regarde par la fenêtre. La souris mange du fromage. Le chien dort dans la cuisine. La voiture passe devant la maison. Le chat boit du lait. La souris court dans la cuisine. Le chien regarde la souris. Le chat et le chien jouent dans la cour. Une croquette tombe sous la table. La maison a une grande fenêtre. Le chat attend la souris près du canapé. L'oiseau chante sur le toit de la maison. L'oiseau mange une graine dans la cour. Le chat regarde l'oiseau par la fenêtre. Le chien aboie après l'oiseau. L'enfant joue avec le chien dans la cour. L'enfant donne une croquette au chien. L'enfant donne du lait au chat. La maman prépare le repas dans la cuisine. Le papa lit le journal sur le canapé. Le chat dort sur les genoux du papa. Le chien dort près de la table. La maman donne du fromage à l'enfant. L'enfant mange du fromage dans la cuisine. Le papa regarde la voiture devant la maison. La voiture roule dans la rue. Le chien court dans la rue après la balle. Le chat court après la souris dans la cuisine. La souris mange une graine sous la table. La souris regarde le chat par le trou. Le chat saute sur le canapé et dort. Le chien saute sur la table et mange le fromage. L'enfant rit quand le chien joue avec la balle. La maman rit quand le chat saute sur la table. Le papa boit du café dans la cuisine. La maman boit du lait avec l'enfant. Le soleil brille sur la maison. Le soleil entre par la fenêtre de la cuisine. Le chat dort au soleil près de la fenêtre. Le chien dort au soleil dans la cour. La pluie tombe sur le toit de la maison. Le chat regarde la pluie par la fenêtre. Le chien rentre dans la maison sous la pluie. L'enfant joue avec la pelote de laine. Le chat joue avec la pelote de laine sur le canapé. La balle roule sous la table. Le chien cherche la balle sous le canapé. Le chat cherche la souris dans la cuisine. L'oiseau regarde le chat sur le toit. La nuit tombe sur la maison. Le chat et la souris dorment dans la maison. Le chien garde la maison pendant la nuit. Le matin le soleil brille sur la cour. Le matin la maman prépare le café dans la cuisine. Le matin le chat boit du lait et le chien mange une croquette. """

# ---- 1) tokenize (sentence by sentence, split l' / d' etc.) ----
sentences = [s.strip().lower() for s in corpus.split(".") if s.strip()]
STOP = {"le", "la", "les", "l", "un", "une", "du", "de", "des", "d", "au", "aux", "se",
        "dans", "sur", "sous", "par", "avec", "et", "a", "à", "après", "près", "pendant",
        "devant", "quand"}
tokenized = [[w for w in re.findall(r"[a-zàâçéèêëîïôûùüœ]+", s.replace("'", " ")) if w not in STOP]
             for s in sentences]

vocab = sorted({w for s in tokenized for w in s})
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"{len(sentences)} phrases, {V} mots uniques\n")

# ---- 2) co-occurrence matrix (window = 2) ----
WINDOW = 3
C = np.zeros((V, V))
for s in tokenized:
    for i, w in enumerate(s):
        for j in range(max(0, i - WINDOW), min(len(s), i + WINDOW + 1)):
            if i != j:
                C[idx[w], idx[s[j]]] += 1

# ---- 3) PPMI + SVD -> dense vectors ----
total = C.sum()
pw = C.sum(1, keepdims=True) / total
pc = C.sum(0, keepdims=True) / total
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((C / total) / (pw * pc))
ppmi = np.nan_to_num(np.maximum(pmi, 0), neginf=0, posinf=0)

U, S, _ = np.linalg.svd(ppmi)
DIM = 5
emb = U[:, :DIM] * S[:DIM]
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9


def similar(word, k=5):
    sims = emb @ emb[idx[word]]
    best = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 2)) for i in best if vocab[i] != word][:k]


for w in ["chat", "chien", "souris", "lait", "cuisine"]:
    print(f"proches de '{w}':", similar(w))

# ---- 4) bigram model: predict next word ----
bigrams = defaultdict(Counter)
for s in tokenized:
    for a, b in zip(s, s[1:]):
        bigrams[a][b] += 1


def next_words(word, k=3):
    total = sum(bigrams[word].values())
    return [(w, round(c / total, 2)) for w, c in bigrams[word].most_common(k)]


print()
for w in ["le", "chat", "mange", "dans"]:
    print(f"après '{w}':", next_words(w))