from gensim.models import FastText

corpus = [
    'Bonjour comment puis je vous aider',
    'Je veux résilier mon abonnement',
    'Mon internet ne fonctionne pas correctement',
    'Je souhaite changer mon forfait',
    'Je veux parler à un conseiller',
    'Je veux savoir comment activer mon compte',
    'Je veux savoir comment désactiver mon compte',
    'Je veux savoir comment changer mon mot de passe',
    'Je veux savoir comment récupérer mon mot de passe',
    'Je veux savoir comment créer un compte'
]
sentences = [sentence.lower().split() for sentence in corpus]

model = FastText(
    sentences,
    vector_size=60,   # 5 is too small to be useful
    window=2,
    min_count=1,
    epochs=50,               # tiny corpus, so train more              # Skip-gram, to match your presentation
    sg=0
)


print(model.wv["bonjour"])
print(model.wv.most_similar("bonjour", topn=5))
