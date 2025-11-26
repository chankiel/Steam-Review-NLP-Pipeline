import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


class NLGGenerator:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        load_dotenv()

        print("[INFO] Initializing HF InferenceClient...")

        self.model_name = model_name
        self.client = InferenceClient(
            model=model_name,
            token=os.getenv("HF_API_KEY")  # Stored in .env
        )

        print(f"[INFO] Using online model: {model_name}")
        print("[INFO] No local download required!")

    # --------------------------------------------------
    # Build Prompt (ABSA supports both scores & labels)
    # --------------------------------------------------
    def build_prompt(self, user_aspects, top_games):
        """
        user_aspects = numeric preferences from UI (-1 to 1)
        top_games = list of top 5 games with their ABSA scores and combined scores
        """
        prompt = f"""
    You are an AI assistant that generates personalized game recommendations.

    ===================
    USER ASPECT PREFERENCES
    ===================
    {json.dumps(user_aspects, indent=2)}

    ===================
    TOP 5 RECOMMENDED GAMES (sorted by combined score - highest first)
    ===================
    The games below are ranked by a combined score of:
    - 10% semantic similarity (how well the game's reviews match your description)
    - 90% aspect similarity (how well the game's aspects align with your preferences)

    Index 1 is the BEST match, Index 2 is second best, and so on.

    {json.dumps(top_games, indent=2)}

    ===================
    TASK
    ===================
    Write a personalized recommendation explaining:
    1. Recommend the TOP game (Index 1) from the list as the best match
    2. Explain why it matches the user's preferences based on aspect alignment and combined score
    3. Briefly mention 1-2 alternative games from the list if the user wants variety
    4. Keep the tone natural and conversational
    5. Final answer must be ONE medium-length paragraph
    """
        return prompt.strip()

    # --------------------------------------------------
    # Generate with HuggingFace Hosted Inference
    # --------------------------------------------------
    def generate(self, prompt, max_tokens=256, temperature=0.7):

        print("[INFO] Sending request to HF Chat Completion API...")

        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message["content"].strip()

        except Exception as e:
            print(f"[ERROR] ChatCompletion failed: {e}")
            raise


# ---------------------------------------------------------
# Example usage (MAIN)
# ---------------------------------------------------------
# if __name__ == "__main__":

#     # Example user preferences (from UI sliders)
#     user_aspects = {
#         "gameplay": 0.9,
#         "graphics": 0.4,
#         "story": 0.2,
#         "performance": 0.8,
#         "value": 0.6
#     }

#     # Example ABSA result using SENTIMENT LABELS
#     game_aspects = {
#         "game_name": "Hades",
#         "aspects": {
#             "gameplay": "positive",
#             "story": "neutral",
#             "graphics": "positive",
#             "performance": "positive",
#             "value": "positive"
#         }
#     }

#     # Example cosine similarity recommender output
#     similar_games = [
#         {"game_name": "Dead Cells", "similarity": 0.89},
#         {"game_name": "Celeste", "similarity": 0.84},
#         {"game_name": "Blasphemous", "similarity": 0.81},
#     ]

#     nlg = NLGGenerator()

#     prompt = nlg.build_prompt(user_aspects, game_aspects, similar_games)

#     print("\n===== PROMPT =====\n")
#     print(prompt)

#     response = nlg.generate(prompt)
#     print("\n===== MODEL RESPONSE =====\n")
#     print(response)
