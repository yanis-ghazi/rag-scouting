import pandas as pd
import json
import os


def clean_value(val, decimals=1):
    """Convertit une valeur en float propre, retourne None si invalide."""
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return None


def process_nba(input_path="data/raw/nba_stats_2024_25.csv",
                output_path="data/processed/nba_processed.json"):
    """
    Lit le CSV NBA, nettoie les données, et génère un JSON
    où chaque joueur a un champ 'text' en langage naturel.
    """
    print("Preprocessing NBA...")
    df = pd.read_csv(input_path)

    players = []

    for _, row in df.iterrows():
        # Nettoyage des valeurs
        name = str(row["PLAYER_NAME"])
        team = str(row["TEAM_ABBREVIATION"])
        age = clean_value(row["AGE"], 0)
        gp = clean_value(row["GP"], 0)
        pts = clean_value(row["PTS"])
        reb = clean_value(row["REB"])
        ast = clean_value(row["AST"])
        stl = clean_value(row["STL"])
        blk = clean_value(row["BLK"])
        tov = clean_value(row["TOV"])
        fg = clean_value(row["FG_PCT"] * 100 if row["FG_PCT"] <= 1 else row["FG_PCT"])
        fg3 = clean_value(row["FG3_PCT"] * 100 if row["FG3_PCT"] <= 1 else row["FG3_PCT"])
        ft = clean_value(row["FT_PCT"] * 100 if row["FT_PCT"] <= 1 else row["FT_PCT"])
        pm = clean_value(row["PLUS_MINUS"])

        # Génération du texte en langage naturel
        text = (
            f"{name}, {age} ans, joue pour {team} (NBA). "
            f"A joué {gp} matchs cette saison. "
            f"Stats par match : {pts} pts, {reb} reb, {ast} ast, "
            f"{stl} stl, {blk} blk, {tov} tov. "
            f"Efficacité : {fg}% FG, {fg3}% 3pts, {ft}% LF. "
            f"+/- moyen : {pm}."
        )

        players.append({
            "id": f"nba_{name.lower().replace(' ', '_')}",
            "sport": "NBA",
            "name": name,
            "team": team,
            "age": age,
            "games_played": gp,
            "pts": pts,
            "reb": reb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "tov": tov,
            "fg_pct": fg,
            "fg3_pct": fg3,
            "ft_pct": ft,
            "plus_minus": pm,
            "text": text
        })

    os.makedirs("data/processed", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(players)} joueurs NBA processés → {output_path}")
    return players


def process_pl(input_path="data/raw/pl_stats_2024_25.csv",
               output_path="data/processed/pl_processed.json"):
    """
    Lit le CSV PL, nettoie les données, et génère un JSON
    où chaque joueur a un champ 'text' en langage naturel.
    """
    print("Preprocessing Premier League...")
    df = pd.read_csv(input_path)

    players = []

    for _, row in df.iterrows():
        name = str(row.get("player", "Unknown"))
        team = str(row.get("team", "Unknown"))
        age_raw = str(row.get("age", "0-0")).split("-")[0]
        age = clean_value(age_raw, 0)
        games = clean_value(row.get("Playing Time_MP", 0), 0)
        minutes = clean_value(row.get("Playing Time_Min", 0), 0)
        goals = clean_value(row.get("Performance_Gls", 0))
        assists = clean_value(row.get("Performance_Ast", 0))
        shots = clean_value(row.get("Standard_Sh", 0))
        shots_on_target = clean_value(row.get("Standard_SoT", 0))
        xg = clean_value(row.get("xg", None))
        position = str(row.get("pos", "Unknown"))
        nation = str(row.get("nation", "Unknown"))

        text = (
            f"{name}, {age} ans, {position}, joue pour {team} (Premier League). "
            f"Nationalité : {nation}. "
            f"A joué {games} matchs ({minutes} minutes). "
            f"Stats : {goals} buts, {assists} passes décisives. "
            f"Tirs : {shots} tentatives, {shots_on_target} cadrés. "
            f"xG : {xg}."
        )

        players.append({
            "id": f"pl_{name.lower().replace(' ', '_')}",
            "sport": "Premier League",
            "name": name,
            "team": team,
            "age": age,
            "position": position,
            "nation": nation,
            "games_played": games,
            "minutes": minutes,
            "goals": goals,
            "assists": assists,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "xg": xg,
            "text": text
        })

    os.makedirs("data/processed", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(players, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(players)} joueurs PL processés → {output_path}")
    return players


if __name__ == "__main__":
    nba_players = process_nba()
    print("\nExemple joueur NBA :")
    print(nba_players[0]["text"])
    print()

    pl_players = process_pl()
    print("\nExemple joueur PL :")
    print(pl_players[0]["text"])