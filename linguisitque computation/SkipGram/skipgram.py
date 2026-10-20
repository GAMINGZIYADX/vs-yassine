from gensim.models import Word2Vec

corpus = [
    ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
    ['I', 'love', 'natural', 'language', 'processing']
]

model = Word2Vec(
    sentences=corpus,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,         # Use skip-gram model
    negative=5    # Number of negative samples
)

print(model.wv.most_similar('fox', topn=5))

