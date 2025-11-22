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
            token=os.getenv("HF_API_KEY")
        )

        print(f"[INFO] Using online model: {model_name}")
        print("[INFO] No local download required!")

    # --------------------------------------------------
    # Build the aspect-enhanced prompt
    # --------------------------------------------------
    def build_prompt(self, user_aspects, game_aspects, similar_games):
        prompt = f"""
You are an AI assistant that generates personalized game recommendations.
Use the user's aspect preferences, ABSA scores, and similar game info
to produce a helpful and personalized recommendation.

===================
USER ASPECT PRIORITIES
===================
{json.dumps(user_aspects, indent=2)}

===================
GAME ASPECT SCORES
===================
{json.dumps(game_aspects, indent=2)}

===================
TOP SIMILAR GAMES (Cosine Similarity)
===================
{json.dumps(similar_games, indent=2)}

===================
TASK
===================
Write a personalized recommendation explaining:

1. Why this game matches the user's preferences.
2. Which aspects align well.
3. Which aspects may not align perfectly.
4. Keep it natural and conversational.
5. One medium-length paragraph.
"""
        return prompt.strip()

    # --------------------------------------------------
    # Generate NLG response (HF Chat Completion API)
    # --------------------------------------------------
    def generate(self, prompt, max_tokens=256, temperature=0.7):

        print("[INFO] Sending request to HF Chat Completion API...")

        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,      # supported
                temperature=temperature     # supported
            )

            return response.choices[0].message["content"].strip()

        except Exception as e:
            print(f"[ERROR] ChatCompletion failed: {e}")
            raise


# ---------------------------------------------------------
# Example usage (MAIN)
# ---------------------------------------------------------
if __name__ == "__main__":

    user_aspects = {
        "gameplay": 0.9,
        "graphics": 0.4,
        "story": 0.2,
        "performance": 0.8,
        "value": 0.6
    }

    game_aspects = {
        "game_name": "Hades",
        "aspects": {
            "gameplay": 0.95,
            "story": 0.5,
            "graphics": 0.6,
            "performance": 0.9
        }
    }

    similar_games = [
        {"game_name": "Dead Cells", "similarity": 0.89},
        {"game_name": "Celeste", "similarity": 0.84},
        {"game_name": "Blasphemous", "similarity": 0.81},
    ]

    nlg = NLGGenerator()

    prompt = nlg.build_prompt(user_aspects, game_aspects, similar_games)
    print("\n===== PROMPT =====\n")
    print(prompt)

    response = nlg.generate(prompt)
    print("\n===== MODEL RESPONSE =====\n")
    print(response)
