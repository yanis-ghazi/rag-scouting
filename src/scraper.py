import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
import soccerdata as sd
import time
import os


def scrape_nba_stats(season="2024-25"):
    print(f"Récupération des stats NBA {season}...")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season"
    )
    df = stats.get_data_frames()[0]
    colonnes_utiles = [
        "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN",
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"
    ]
    df = df[colonnes_utiles]
    df = df[df["GP"] >= 10]
    df = df.sort_values("PTS", ascending=False)
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/nba_stats_2024_25.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ {len(df)} joueurs NBA sauvegardés dans {output_path}")
    return df


def scrape_pl_stats(season="2025"):
    print(f"Récupération des stats Premier League {season}...")
    print("(Première fois : peut prendre 1-2 minutes)")
    try:
        fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)

        df_standard = fbref.read_player_season_stats(stat_type="standard").reset_index()
        df_shooting = fbref.read_player_season_stats(stat_type="shooting").reset_index()

        # Aplatir les colonnes multi-index
        df_standard.columns = [
            f"{a}_{b}".strip("_") if b else a
            for a, b in df_standard.columns
        ] if isinstance(df_standard.columns, pd.MultiIndex) else df_standard.columns

        df_shooting.columns = [
            f"{a}_{b}".strip("_") if b else a
            for a, b in df_shooting.columns
        ] if isinstance(df_shooting.columns, pd.MultiIndex) else df_shooting.columns

        # Affiche les colonnes pour debug
        print("Colonnes standard:", df_standard.columns.tolist()[:10])
        print("Colonnes shooting:", df_shooting.columns.tolist()[:10])

        # Fusion
        df = df_standard.merge(df_shooting, on=["player", "team"], how="left", suffixes=("", "_shot"))

        # Filtre joueurs avec au moins 5 matchs
        if "games" in df.columns:
            df = df[df["games"] >= 5]
        elif "MP" in df.columns:
            df = df[df["MP"] >= 5]

        os.makedirs("data/raw", exist_ok=True)
        output_path = "data/raw/pl_stats_2024_25.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ {len(df)} joueurs PL sauvegardés dans {output_path}")
        return df

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # NBA
    df_nba = scrape_nba_stats()
    print("\n--- Aperçu NBA ---")
    print(df_nba.head(5).to_string())

    # Premier League
    df_pl = scrape_pl_stats()
    if df_pl is not None:
        print("\n--- Aperçu Premier League ---")
        print(df_pl.head(5).to_string())