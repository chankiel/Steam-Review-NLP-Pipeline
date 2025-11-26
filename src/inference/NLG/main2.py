# main2.py
import pandas as pd
import numpy as np
from tqdm import tqdm
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


def calculate_aspect_similarity(user_aspects, game_aspects):
    """Calculate cosine similarity between user preferences and game ABSA."""
    user_vec = np.array([user_aspects[asp] for asp in ASPECTS])
    game_vec = np.array([game_aspects[asp] for asp in ASPECTS])
    
    # Cosine similarity
    dot_product = np.dot(user_vec, game_vec)
    norm_user = np.linalg.norm(user_vec)
    norm_game = np.linalg.norm(game_vec)
    
    if norm_user == 0 or norm_game == 0:
        return 0.0
    
    return dot_product / (norm_user * norm_game)


def main():
    print("===== INITIALIZING PIPELINE =====")

    # Load ABSA sentiment table
    absa_df = load_absa()

    # Load Pinecone + Sentence Embedding
    pm = PineconeManager()

    # Load Llama NLG Generator
    nlg = NLGGenerator()

    # 1. Ask user for aspect preferences
    user_aspects = get_user_preferences()

    # 2. Get user review/comment
    user_comment = get_user_comment()

    # 3. Calculate ABSA similarity for ALL games
    print("\n===== CALCULATING ASPECT SIMILARITY FOR ALL GAMES =====")
    games_with_absa_sim = []
    
    for idx, row in tqdm(absa_df.iterrows(), total=len(absa_df), desc="Processing games"):
        game_name = row["app_name"]
        game_aspects = {asp: float(row[asp]) for asp in ASPECTS}
        
        aspect_sim = calculate_aspect_similarity(user_aspects, game_aspects)
        
        games_with_absa_sim.append({
            "game_name": game_name,
            "aspect_similarity": aspect_sim,
            "aspects": game_aspects
        })

    # 4. Sort by aspect similarity and take top 20
    games_with_absa_sim.sort(key=lambda x: x["aspect_similarity"], reverse=True)
    top_20_absa = games_with_absa_sim[:20]

    print("\n===== TOP 20 GAMES BY ASPECT SIMILARITY =====")
    for i, g in enumerate(top_20_absa, 1):
        print(f"{i:2d}. {g['game_name']}: {g['aspect_similarity']:.4f}")

    # 5. Query Pinecone for each of the top 20 games
    print("\n===== QUERYING PINECONE FOR TOP 20 GAMES =====")
    games_with_pinecone = []
    
    for game in tqdm(top_20_absa, desc="Querying Pinecone"):
        game_name = game["game_name"]
        
        # Query Pinecone with user comment to get similarity score
        results = pm.query_text(user_comment, top_k=100)  # Get more to ensure we find the game
        
        # Find this specific game in results
        pinecone_score = 0.0
        for r in results:
            if r["metadata"]["app_name"] == game_name:
                pinecone_score = float(r["score"])
                break
        
        games_with_pinecone.append({
            "game_name": game_name,
            "aspect_similarity": game["aspect_similarity"],
            "pinecone_score": pinecone_score,
            "aspects": game["aspects"]
        })

    # 6. Sort by Pinecone score and take top 5
    games_with_pinecone.sort(key=lambda x: x["pinecone_score"], reverse=True)
    top_5_games = games_with_pinecone[:5]

    print("\n===== TOP 5 GAMES BY PINECONE SIMILARITY =====")
    for i, g in enumerate(top_5_games, 1):
        print(f"{i}. {g['game_name']}")
        print(f"   - Aspect Similarity: {g['aspect_similarity']:.4f}")
        print(f"   - Pinecone Score: {g['pinecone_score']:.4f}")

    # 7. Build NLG prompt with top 5 games
    prompt = nlg.build_prompt(
        user_aspects=user_aspects,
        top_games=top_5_games
    )

    # 8. Generate recommendation
    print("\n===== AI RECOMMENDATION =====")
    response = nlg.generate(prompt)
    print(response)


if __name__ == "__main__":
    main()