# main3.py
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

    # 3. Query Pinecone to get ALL games with similarity scores
    print("\n===== QUERYING PINECONE FOR ALL GAMES =====")
    # Query with high top_k to get as many games as possible
    # Pinecone typically allows up to 10,000
    pinecone_results = pm.query_text(user_comment, top_k=10000)
    
    print(f"[INFO] Retrieved {len(pinecone_results)} games from Pinecone")
    
    # Store Pinecone scores in a dict
    pinecone_scores = {}
    for r in pinecone_results:
        game_name = r["metadata"]["app_name"]
        pinecone_scores[game_name] = float(r["score"])

    # 4. Calculate ABSA similarity for ALL games
    print("\n===== CALCULATING ASPECT SIMILARITY FOR ALL GAMES =====")
    absa_scores = {}
    
    for idx, row in tqdm(absa_df.iterrows(), total=len(absa_df), desc="ABSA similarity"):
        game_name = row["app_name"]
        game_aspects = {asp: float(row[asp]) for asp in ASPECTS}
        
        aspect_sim = calculate_aspect_similarity(user_aspects, game_aspects)
        absa_scores[game_name] = {
            "similarity": aspect_sim,
            "aspects": game_aspects
        }

    # 5. Combine scores: 0.5 * pinecone + 0.5 * absa
    print("\n===== CALCULATING COMBINED SCORES =====")
    combined_games = []
    
    # Use ALL games from ABSA (not just common games)
    print(f"[INFO] Processing {len(absa_scores)} games from ABSA dataset")
    
    for game_name in tqdm(absa_scores.keys(), desc="Combining scores"):
        # Get Pinecone score (0 if not found)
        pinecone_score = pinecone_scores.get(game_name, 0.0)
        
        absa_score = absa_scores[game_name]["similarity"]
        combined_score = 0.5 * pinecone_score + 0.5 * absa_score
        
        combined_games.append({
            "game_name": game_name,
            "combined_score": combined_score,
            "pinecone_score": pinecone_score,
            "aspect_similarity": absa_score,
            "aspects": absa_scores[game_name]["aspects"]
        })

    # 6. Sort by combined score and take top 5
    combined_games.sort(key=lambda x: x["combined_score"], reverse=True)
    top_5_games = combined_games[:5]

    print("\n===== TOP 5 GAMES BY COMBINED SCORE =====")
    for i, g in enumerate(top_5_games, 1):
        print(f"{i}. {g['game_name']}")
        print(f"   - Combined Score: {g['combined_score']:.4f}")
        print(f"   - Pinecone Score: {g['pinecone_score']:.4f}")
        print(f"   - Aspect Score: {g['aspect_similarity']:.4f}")

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