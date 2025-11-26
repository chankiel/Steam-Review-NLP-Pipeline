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
    def build_prompt(self, user_aspects, game_aspects, similar_games):
        """
        user_aspects = numeric preferences from UI (0–1)
        game_aspects = ABSA output (can be numeric OR 'positive/negative/neutral')
        similar_games = list of cosine similarity matches
        """
        prompt = f"""
You are an AI assistant that generates personalized game recommendations.
Use:
- The user's aspect preferences
- The ABSA sentiment scores (positive/negative/neutral or numeric 0–1)
- The computed similar-game list

===================
USER ASPECT PRIORITIES
===================
{json.dumps(user_aspects, indent=2)}

===================
GAME ASPECT SENTIMENT (ABSA)
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
3. Which aspects might not align perfectly.
4. Keep the tone natural and conversational.
5. Final answer must be ONE medium-length paragraph.
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
