import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


def init_components():
    """Initialise le modèle d'embeddings, ChromaDB et le client Groq."""
    print("Initialisation des composants...")

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("players")

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("✅ Composants initialisés !")
    return model, collection, groq_client


def extract_filters(question, groq_client):
    """
    Utilise Groq pour extraire les filtres numériques
    et le sport concerné depuis la question.
    """
    prompt = f"""Tu es un assistant qui extrait des filtres de recherche depuis une question sur des stats sportives.

Question : "{question}"

Extrais les informations suivantes en JSON strict (sans markdown, sans explication) :
{{
  "sport": "NBA" ou "Premier League" ou "both",
  "age_max": nombre ou null,
  "age_min": nombre ou null,
  "pts_min": nombre ou null,
  "ast_min": nombre ou null,
  "ast_max": nombre ou null,
  "reb_min": nombre ou null,
  "tov_max": nombre ou null,
  "goals_min": nombre ou null,
  "assists_min": nombre ou null,
  "query_text": "reformulation courte de la question pour la recherche sémantique"
}}

Réponds UNIQUEMENT avec le JSON, rien d'autre."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )

    raw = response.choices[0].message.content.strip()

    try:
        filters = json.loads(raw)
    except json.JSONDecodeError:
        # Si le LLM n'a pas respecté le format JSON on retourne des filtres vides
        filters = {"sport": "both", "query_text": question}

    return filters


def search_players(question, model, collection, filters, n_results=10):
    """
    Cherche les joueurs pertinents dans ChromaDB.
    Combine filtres metadata + recherche sémantique.
    """
    query_text = filters.get("query_text", question)
    sport = filters.get("sport", "both")

    # Génère l'embedding de la question
    query_embedding = model.encode([query_text]).tolist()

    # Filtre par sport si précisé
    where_filter = None
    if sport == "NBA":
        where_filter = {"sport": "NBA"}
    elif sport == "Premier League":
        where_filter = {"sport": "Premier League"}

    # Recherche dans ChromaDB
    if where_filter:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter
        )
    else:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

    # Récupère les textes et métadonnées
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


def apply_numeric_filters(documents, metadatas, filters, all_players_json):
    """
    Applique les filtres numériques sur les joueurs récupérés.
    Si trop peu de résultats, élargit la recherche depuis le JSON complet.
    """
    sport = filters.get("sport", "both")

    # Charge tous les joueurs pour les filtres numériques précis
    all_players = []
    if sport in ["NBA", "both"]:
        with open("data/processed/nba_processed.json", "r", encoding="utf-8") as f:
            all_players += json.load(f)
    if sport in ["Premier League", "both"]:
        with open("data/processed/pl_processed.json", "r", encoding="utf-8") as f:
            all_players += json.load(f)

    filtered = []
    for p in all_players:
        # Filtres NBA
        if filters.get("pts_min") and p.get("pts"):
            if p["pts"] < filters["pts_min"]:
                continue
        if filters.get("ast_min") and p.get("ast"):
            if p["ast"] < filters["ast_min"]:
                continue
        if filters.get("ast_max") and p.get("ast"):
            if p["ast"] > filters["ast_max"]:
                continue
        if filters.get("tov_max") and p.get("tov"):
            if p["tov"] > filters["tov_max"]:
                continue
        if filters.get("reb_min") and p.get("reb"):
            if p["reb"] < filters["reb_min"]:
                continue

        # Filtres PL
        if filters.get("goals_min") and p.get("goals"):
            if p["goals"] < filters["goals_min"]:
                continue
        if filters.get("assists_min") and p.get("assists"):
            if p["assists"] < filters["assists_min"]:
                continue

        # Filtre age
        if filters.get("age_max") and p.get("age"):
            if p["age"] > filters["age_max"]:
                continue
        if filters.get("age_min") and p.get("age"):
            if p["age"] < filters["age_min"]:
                continue

        filtered.append(p)

    return filtered


def generate_answer(question, players, groq_client, filters=None):
    if not players:
        return "Aucun joueur trouvé correspondant à ces critères."

    if filters is None:
        filters = {}

    sort_key = "pts"
    if players and players[0].get("sport") == "Premier League":
        if filters.get("assists_min") or "passeur" in question.lower() or "assist" in question.lower():
            sort_key = "assists"
        else:
            sort_key = "goals"
    else:
        if filters.get("ast_min") or "assist" in question.lower() or "passe" in question.lower():
            sort_key = "ast"
        elif filters.get("reb_min") or "rebond" in question.lower():
            sort_key = "reb"

    players_sorted = sorted(players, key=lambda x: x.get(sort_key) or 0, reverse=True)
    players_text = "\n".join([p["text"] for p in players_sorted[:15]])

    prompt = f"""Tu es un expert scout sportif. Réponds en français à la question suivante en te basant UNIQUEMENT sur les données fournies.

Question : {question}

Joueurs disponibles :
{players_text}

Instructions :
- Réponds de façon claire et structurée
- Cite les stats précises des joueurs
- Si plusieurs joueurs correspondent, classe-les du meilleur au moins bon
- Si aucun joueur ne correspond parfaitement, dis-le clairement
- Réponds en français"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def ask(question, model, collection, groq_client):
    """
    Fonction principale : pose une question et obtient une réponse.
    """
    print(f"\n🔍 Question : {question}")
    print("Analyse en cours...")

    # 1. Extraire les filtres
    filters = extract_filters(question, groq_client)
    print(f"Filtres détectés : {filters}")

    # 2. Chercher dans ChromaDB
    documents, metadatas = search_players(question, model, collection, filters)

    # 3. Appliquer les filtres numériques
    filtered_players = apply_numeric_filters(documents, metadatas, filters, None)
    print(f"{len(filtered_players)} joueurs après filtrage numérique")

    # 4. Générer la réponse
    answer = generate_answer(question, filtered_players, groq_client, filters)

    return answer


if __name__ == "__main__":
    model, collection, groq_client = init_components()

    questions = [
        "Quel joueur NBA sous 25 ans a le plus d'assists cette saison ?",
        "Trouve moi un meneur NBA avec plus de 8 assists et moins de 3 turnovers",
        "Quel est le meilleur buteur de Premier League cette saison ?",
        "Quel est le meilleur passeur décisif de Premier League cette saison ?"
    ]

    for question in questions:
        answer = ask(question, model, collection, groq_client)
        print(f"\n💬 Réponse :\n{answer}")
        print("\n" + "="*60)