import spacy
# Charger le model francasi
nlp = spacy.load('fr_core_news_md')

# Phrases a analyser
textes = [
    "J'ai, euh, acheté... un nouveau téléphone, mais, hum, je n'aime pas, vraiment, la couleur. En fait... je pensais que ça serait... bleu? Mais, en fait, c'est plus proche de gris.",
    "Le produit que j'ai reçu est défectueux !! Franchement, je suis très déçu de la qualité... Cela n'a rien à voir avec la description sur le site internet et maintenant je dois attendre pour obtenir un remboursement... Bref, vraiment pas satisfait !!",
    "Salut, j'espère que tout va bien !! Je voulais te parler du produit XYZ que tu m'as recommandé. Euh... Je pense que c'est un bon produit, mais j'ai eu quelques soucis au début . Je te dis ça juste pour que tu saches, voilà. À plus tard.",
    "Bon alors, je viens d'acheter un nouveau portable sur Amazon, et je dois dire que... wow, je suis super content . Franchement, c'est génial. Cependant, il y a un petit problème : la batterie se décharge très vite . A part ça, tout est parfait.",
    "L'hôtel était situé près de la plage. Mais... j'ai trouvé que les chambres étaient très, très petites. Hum... Les photos en ligne montrent des chambres beaucoup plus grandes, et je m'attendais à mieux. Le personnel était correct, mais pas très sympathique.",
    "Ce nouveau film est INCROYABLE!!! Les acteurs sont géniaux et l'histoire est captivante. Mais... parfois, le rythme est un peu lent. Par contre, la fin est juste sublime!!!",
    "Ce produit a littéralement changé ma vie !!! Avant, je passais des heures à faire ça manuellement, et maintenant, tout est automatisé !! Sérieusement, c'est incroyable.",
    "Le match d'hier soir était vraiment impressionnant!! Les deux équipes ont joué avec une énergie incroyable, mais malheureusement l'équipe locale a perdu ... Mais bon, ça reste du sport, on ne gagne pas à tous les coups.",
    "Je vous écris pour exprimer ma déception concernant votre service client. J'ai attendu 30 minutes au téléphone pour parler à un représentant, et quand j'ai enfin eu quelqu'un, il n'a pas pu résoudre mon problème. C'est inacceptable ! J'espère que vous allez prendre les mesures nécessaires pour corriger cela.",
    "Le film que j'ai regardé hier soir sur Netflix était juste génial ! Les acteurs sont supers et l'intrigue était captivante du début à la fin... Par contre, la musique de fond était un peu forte à certains moments. Mais sinon, c'était top !",
]

# Analyse NLP
for i, doc in enumerate(nlp.pipe(textes), start=1):
    print(f"\n=== TEXTE {i} ===")
    print("MOT\t\tPOS")
    for token in doc:
        print(f"{token.text:<12}{token.pos_:<12}")
