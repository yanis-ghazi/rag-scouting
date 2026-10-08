import os
import json
from functools import lru_cache

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_CANDIDATES = 15

RANGE_RULES = [
    ("pts_min", "pts", "min"),
    ("ast_min", "ast", "min"),
    ("ast_max", "ast", "max"),
    ("reb_min", "reb", "min"),
    ("tov_max", "tov", "max"),
    ("fg3a_min", "fg3a", "min"),
    ("goals_min", "goals", "min"),
    ("assists_min", "assists", "min"),
    ("age_min", "age", "min"),
    ("age_max", "age", "max"),
]

FILTER_PROMPT = """Tu extrais des critères de recherche depuis une question sur des stats sportives.

Question : "__Q__"

Réponds uniquement avec un JSON strict :
{
  "sport": "NBA" ou "Premier League" ou "both",
  "team": abréviation NBA (ex: "GSW", "LAL") ou nom du club de Premier League comme écrit par FBref (ex: "Arsenal", "Manchester City"), sinon null,
    "position": "DF", "MF", "FW" ou "GK" uniquement si la question mentionne explicitement un poste (défenseur, milieu, attaquant, gardien), sinon null,
  "sort_by": statistique à classer ("pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct", "plus_minus", "goals", "assists", "shots", "shots_on_target" ou "age"), uniquement si la question demande explicitement un classement ou un extrême (le plus, le meilleur, le moins, top). Pour un profil ou un style de joueur (polyvalent, créatif, complet...), mets null, "sort_order": "desc" pour "le plus" ou "meilleur", "asc" pour "le moins" ou "le pire",
  "age_max": nombre ou null (« sous N ans » signifie strictement moins de N, donc mets N-1),
  "age_min": nombre ou null,
  "pts_min": nombre ou null,
  "ast_min": nombre ou null,
  "ast_max": nombre ou null,
  "reb_min": nombre ou null,
  "tov_max": nombre ou null,
  "fg3a_min": nombre ou null,
  "goals_min": nombre ou null,
  "assists_min": nombre ou null,
  "query_text": "reformulation courte de la question pour la recherche sémantique"
}"""


def init_components():
    print("Initialisation des composants...")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("players")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Composants initialisés")
    return model, collection, groq_client


@lru_cache(maxsize=None)
def load_players(sport):
    paths = {
        "NBA": "data/processed/nba_processed.json",
        "Premier League": "data/processed/pl_processed.json",
    }
    selected = paths.keys() if sport == "both" else [sport]
    players = []
    for key in selected:
        with open(paths[key], "r", encoding="utf-8") as f:
            players += json.load(f)
    return players


def extract_filters(question, groq_client):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": FILTER_PROMPT.replace("__Q__", question)}],
        temperature=0,
        max_tokens=2000,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        filters = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        print("JSON invalide :", raw)
        filters = {"sport": "both", "query_text": question}
    if filters.get("sport") not in ("NBA", "Premier League"):
        filters["sport"] = "both"
    filters.setdefault("query_text", question)
    return filters


def has_constraints(filters):
    keys = ["team", "position", "sort_by"] + [rule[0] for rule in RANGE_RULES]
    return any(filters.get(k) is not None for k in keys)


def apply_filters(filters):
    team = filters.get("team")
    position = filters.get("position")
    result = []

    for p in load_players(filters["sport"]):
        if team and team.lower() not in p["team"].lower():
            continue
        if position and position.upper() not in str(p.get("position", "")).upper():
            continue

        valid = True
        for key, field, kind in RANGE_RULES:
            limit = filters.get(key)
            if limit is None:
                continue
            value = p.get(field)
            if value is None or (kind == "min" and value < limit) or (kind == "max" and value > limit):
                valid = False
                break
        if valid:
            result.append(p)

    return result


def rank_by_stat(players, filters):
    sort_by = filters["sort_by"]
    ranked = [p for p in players if p.get(sort_by) is not None]
    ranked.sort(key=lambda p: p[sort_by], reverse=filters.get("sort_order") != "asc")
    return ranked


def rank_by_similarity(players, query_text, model, collection):
    if len(players) <= 1:
        return players
    by_id = {p["id"]: p for p in players}
    stored = collection.get(ids=list(by_id), include=["embeddings"])
    embeddings = np.array(stored["embeddings"])
    query = model.encode([query_text])[0]
    scores = embeddings @ query / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query) + 1e-9)
    order = np.argsort(-scores)
    return [by_id[stored["ids"][i]] for i in order]


def vector_search(query_text, sport, model, collection):
    params = {
        "query_embeddings": model.encode([query_text]).tolist(),
        "n_results": MAX_CANDIDATES,
    }
    if sport in ("NBA", "Premier League"):
        params["where"] = {"sport": sport}
    ids = collection.query(**params)["ids"][0]
    by_id = {p["id"]: p for p in load_players("both")}
    return [by_id[i] for i in ids if i in by_id]


def retrieve(question, filters, model, collection):
    query_text = filters["query_text"]
    if not has_constraints(filters):
        print("Mode : recherche vectorielle")
        return vector_search(query_text, filters["sport"], model, collection)

    candidates = apply_filters(filters)
    if filters.get("sort_by"):
        print("Mode : filtres + tri par statistique")
        return rank_by_stat(candidates, filters)

    print("Mode : filtres + reclassement sémantique")
    return rank_by_similarity(candidates, query_text, model, collection)


def generate_answer(question, players, groq_client):
    if not players:
        return "Aucun joueur ne correspond à ces critères dans les données disponibles."

    players_text = "\n".join(p["text"] for p in players[:MAX_CANDIDATES])

    prompt = f"""Tu es un expert scout sportif. Réponds en français à la question en te basant uniquement sur les données fournies.

Question : {question}

Joueurs, déjà filtrés et classés du plus pertinent au moins pertinent :
{players_text}

Consignes :
- Réponds de façon claire et structurée
- Cite les statistiques précises
- Respecte l'ordre fourni pour le classement
- Si les données ne permettent pas de répondre, dis-le clairement"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2500,
        reasoning_effort="low",
    )
    return response.choices[0].message.content


def ask(question, model, collection, groq_client):
    print(f"\nQuestion : {question}")
    filters = extract_filters(question, groq_client)
    print(f"Filtres détectés : {filters}")

    players = retrieve(question, filters, model, collection)
    print(f"{len(players)} joueurs retenus")

    return generate_answer(question, players, groq_client)


if __name__ == "__main__":
    model, collection, groq_client = init_components()

    questions = [
        "Quel joueur NBA sous 25 ans a le plus d'assists cette saison ?",
        "Trouve moi un meneur NBA avec plus de 8 assists et moins de 3 turnovers",
        "Quel joueur NBA a le meilleur pourcentage à 3 points chez les GSW ?",
        "Quel défenseur de Premier League a marqué le plus de buts ?",
        "Quel est le meilleur passeur décisif de Premier League ?",
        "Trouve moi un joueur NBA polyvalent qui score, passe et rebondit",
    ]

    for q in questions:
        print(f"\nRéponse :\n{ask(q, model, collection, groq_client)}")
        print("=" * 60)