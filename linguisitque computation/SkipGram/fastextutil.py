#import fasttext.util
#fasttext.util.download_model('fr')  # Download the French model
import fasttext


model = fasttext.load_model('cc.fr.300.bin')  # Load the French model
mots = model.get_nearest_neighbors('bonjour', k=18)  # Get the 18 nearest neighbors of the word 'bonjour'
for score, mot in mots:
    print(mot, round(score, 4))  # Print the word and its similarity score
