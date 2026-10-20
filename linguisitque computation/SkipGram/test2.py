from pathlib import Path
from gensim.models.fasttext import load_facebook_vectors

# Load the local French FastText model (same folder as this script)
wv = load_facebook_vectors(Path(__file__).parent / "cc.fr.300.bin")

# Nearest neighbors
print(wv.most_similar("roi"))

# Word vector shape
print(wv["roi"].shape)  # (300,)

# Similarity between two words
print(wv.similarity("roi", "reine"))

# Analogy: roi - homme + femme ≈ reine
print(wv.most_similar(positive=["roi", "femme"], negative=["homme"], topn=3))

# Out-of-vocabulary word (FastText still gives a vector thanks to subwords)
print(wv["roiiiii"].shape)