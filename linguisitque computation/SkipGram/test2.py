from pathlib import Path
from gensim.models.fasttext import load_facebook_vectors

wv = load_facebook_vectors(Path(__file__).parent / "cc.fr.300.bin")

print(wv.most_similar("roi"))

print(wv["roi"].shape)  # (300,)

# Similarity between two words
print(wv.similarity("roi", "reine"))

print(wv.most_similar(positive=["roi", "femme"], negative=["homme"], topn=3))

print(wv["roiiiii"].shape)