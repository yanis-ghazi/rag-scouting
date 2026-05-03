# RAG Scouting Sport

Outil de scouting intelligent utilisant le RAG (Retrieval-Augmented Generation) pour répondre à des questions en langage naturel sur les statistiques de joueurs NBA et Premier League.

## Exemples de questions

- "Quel joueur NBA sous 25 ans a le plus d'assists cette saison ?"
- "Trouve moi un meneur NBA avec plus de 8 assists et moins de 3 turnovers"
- "Quel est le meilleur buteur de Premier League cette saison ?"
- "Trouve moi un défenseur de PL de moins de 23 ans avec plus de 5 buts"

## Architecture

Question utilisateur
↓
Groq LLM extrait les filtres numériques
↓
ChromaDB recherche par similarité vectorielle
↓
Filtrage numérique sur les métadonnées
↓
Groq LLM génère la réponse finale

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Interface | Gradio |
| Données foot | FBref via soccerdata |
| Données basket | NBA API officielle |

## Données

- 508 joueurs NBA — saison 2024/25 (stats par match : pts, reb, ast, stl, blk, tov, FG%, 3P%, FT%)
- 539 joueurs Premier League — saison 2024/25 (buts, passes décisives, tirs, minutes jouées)

## Installation

```bash
git clone https://github.com/TON_USERNAME/rag-scouting.git
cd rag-scouting

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

# Ajouter la clé API dans .env
cp .env.example .env

python src/scraper.py
python src/preprocessor.py
python src/indexer.py
python app.py
```

## Structure du projet

rag-scouting/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── scraper.py
│   ├── preprocessor.py
│   ├── indexer.py
│   └── rag_engine.py
├── app.py
├── requirements.txt
└── README.md

## Auteur

Etudiant en 4ème année d'école d'ingénieur, spécialisation data science appliquée au sport.