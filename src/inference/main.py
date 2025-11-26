# main.py
import pandas as pd
import numpy as np
from PineconeManager import PineconeManager
from NLGGenerator import NLGGenerator

ASPECTS = [
    "gameplay", "graphics", "performance", "story",
    "controls", "audio", "price", "multiplayer"
]

def normalize_rating(r):
    """Normalize from 0–10 into -1 to 1."""
    return (r - 5) / 5


def get_user_preferences():
    print("\n===== USER ASPECT RATING INPUT =====")

    user_aspects = {}
    for asp in ASPECTS:
        while True:
            try:
                val = float(input(f"Rate {asp} (0–10): "))
                if 0 <= val <= 10:
                    user_aspects[asp] = normalize_rating(val)
                    break
                else:
                    print("⚠ Please enter a number between 0 and 10.")
            except:
                print("⚠ Invalid number. Try again.")

    return user_aspects


def get_user_comment():
    print("\n===== USER REVIEW INPUT =====")
    return input("Write your review/comment: ")


def load_absa():
    print("\n===== LOADING ABSA DATA =====")
    df = pd.read_csv("absa_game.csv")
    print(f"[INFO] Loaded {len(df)} ABSA rows.")
    return df


def get_game_absa(absa_df, game_name):
    """Look up ABSA row for a specific game."""
    row = absa_df[absa_df["app_name"] == game_name]

    if row.empty:
        return None  # missing ABSA
    row = row.iloc[0]

    return {
        "game_name": game_name,
        "aspects": {asp: float(row[asp]) for asp in ASPECTS}
    }


def main():
    print("===== INITIALIZING PIPELINE =====")

    # Load ABSA sentiment table
    absa_df = load_absa()

    # Load Pinecone + Sentence Embedding
    pm = PineconeManager()

    # Load Llama NLG Generator
    nlg = NLGGenerator()

    # -------------------------------------------------------------
    # 1. Ask user for aspect preferences
    # -------------------------------------------------------------
    user_aspects = get_user_preferences()

    # -------------------------------------------------------------
    # 2. Get user review/comment
    # -------------------------------------------------------------
    user_comment = get_user_comment()

    # -------------------------------------------------------------
    # 3. Query Pinecone for similar games
    # -------------------------------------------------------------
    print("\n===== QUERYING PINECONE =====")
    results = pm.query_text(user_comment, top_k=5)

    if len(results) == 0:
        print("[ERROR] No similar games found.")
        return

    print("\nTop Matches:")
    similar_games = []
    for r in results:
        print(f"- {r['metadata']['app_name']}  (score={r['score']:.4f})")
        similar_games.append({
            "game_name": r["metadata"]["app_name"],
            "similarity": float(r["score"])
        })

    # Choose the first (best match)
    best_game_name = results[0]["metadata"]["app_name"]
    print(f"\n[INFO] Best match: {best_game_name}")

    # -------------------------------------------------------------
    # 4. Retrieve ABSA for the best‐matched game
    # -------------------------------------------------------------
    game_absa = get_game_absa(absa_df, best_game_name)

    if game_absa is None:
        print("[WARNING] No ABSA for this game. Using neutral values.")
        game_absa = {
            "game_name": best_game_name,
            "aspects": {asp: 0.0 for asp in ASPECTS}
        }

    # -------------------------------------------------------------
    # 5. Build NLG prompt
    # -------------------------------------------------------------
    prompt = nlg.build_prompt(
        user_aspects=user_aspects,
        game_aspects=game_absa,
        similar_games=similar_games
    )

    # -------------------------------------------------------------
    # 6. Generate recommendation
    # -------------------------------------------------------------
    print("\n===== AI RECOMMENDATION =====")
    response = nlg.generate(prompt)
    print(response)


if __name__ == "__main__":
    main()
