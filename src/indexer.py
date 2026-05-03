import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


# Modèle d'embeddings - léger et performant, pas besoin de GPU
MODEL_NAME = "all-MiniLM-L6-v2"


def load_players(nba_path="data/processed/nba_processed.json",
                 pl_path="data/processed/pl_processed.json"):
    """Charge les joueurs NBA et PL depuis les fichiers JSON."""
    players = []

    with open(nba_path, "r", encoding="utf-8") as f:
        players += json.load(f)

    with open(pl_path, "r", encoding="utf-8") as f:
        players += json.load(f)

    print(f"✅ {len(players)} joueurs chargés au total")
    return players


def build_index(players, db_path="chroma_db"):
    """
    Génère les embeddings et les stocke dans ChromaDB.
    """
    print(f"Chargement du modèle d'embeddings : {MODEL_NAME}")
    print("(Première fois : téléchargement ~90MB, soyez patient...)")

    # Chargement du modèle sentence-transformers
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Modèle chargé !")

    # Initialisation de ChromaDB en mode local (fichiers sur disque)
    client = chromadb.PersistentClient(path=db_path)

    # Supprime la collection si elle existe déjà (pour éviter les doublons)
    try:
        client.delete_collection("players")
        print("Collection existante supprimée")
    except:
        pass

    # Création de la collection
    collection = client.create_collection(
        name="players",
        metadata={"hnsw:space": "cosine"}  # Similarité cosinus
    )

    # Préparation des données par batches de 100
    # (pour ne pas surcharger la mémoire)
    batch_size = 100
    total = len(players)

    print(f"Indexation de {total} joueurs...")

    for i in range(0, total, batch_size):
        batch = players[i:i + batch_size]

        # Les textes à transformer en embeddings
        texts = [p["text"] for p in batch]

        # Les IDs uniques
        ids = [p["id"] for p in batch]

        # Les métadonnées (stats structurées) stockées à côté
        metadatas = [{
            "name": p["name"],
            "sport": p["sport"],
            "team": p["team"],
            "age": float(p["age"]) if p["age"] else 0.0,
        } for p in batch]

        # Génération des embeddings
        embeddings = model.encode(texts).tolist()

        # Ajout dans ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"  {min(i + batch_size, total)}/{total} joueurs indexés...")

    print(f"✅ Index ChromaDB créé dans '{db_path}/'")
    return collection


def test_index(db_path="chroma_db"):
    """Test rapide pour vérifier que l'index fonctionne."""
    print("\n--- Test de l'index ---")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("players")

    # Test 1 : joueur NBA avec beaucoup d'assists
    query = "meneur NBA avec beaucoup d'assists"
    embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=3,
        where={"sport": "NBA"}
    )

    print(f"\nRequête : '{query}'")
    for doc in results["documents"][0]:
        print(f"  → {doc[:100]}...")

    # Test 2 : attaquant PL qui marque beaucoup
    query2 = "attaquant Premier League meilleur buteur"
    embedding2 = model.encode([query2]).tolist()

    results2 = collection.query(
        query_embeddings=embedding2,
        n_results=3,
        where={"sport": "Premier League"}
    )

    print(f"\nRequête : '{query2}'")
    for doc in results2["documents"][0]:
        print(f"  → {doc[:100]}...")


if __name__ == "__main__":
    # 1. Charger les joueurs
    players = load_players()

    # 2. Construire l'index
    collection = build_index(players)

    # 3. Tester
    test_index()