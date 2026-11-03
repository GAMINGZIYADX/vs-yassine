# importation
import json
from pathlib import Path
#chargement du fichier json
json_path = Path(__file__).resolve().parent / "ia_chat_data (2).json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

#liste des intentions uniques
classes = []
for intent in data["intents"]:
    if intent["intent"] not in classes:
        classes.append(intent["intent"])

#affichage de toutes les intentions
for intent in classes:
    print(intent)

#affichage des questions
for intent in data["intents"]:
        print(f"Intent: {intent['intent']}")
        for questions in intent["questions"]:
            print(f" - Question: {questions}")
#affichage des réponses
for intent in data["intents"]:
        print(f"Intent: {intent['intent']}")
        for responses in intent["responses"]:
            print(f" - Response: {responses}")
