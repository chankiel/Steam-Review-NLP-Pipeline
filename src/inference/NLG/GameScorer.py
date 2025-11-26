import numpy as np

class GameScorer:
    def __init__(self):
        pass

    @staticmethod
    def cosine_similarity(embedding_user, embedding_game):
        """
        Compute cosine similarity between two embeddings.
        expected data format (MAY CHANGE LATER)
        Inputs:
            embedding_user: np.array or list, shape (dim,)
            embedding_game: np.array or list, shape (dim,)
        Output:
            cosine similarity float in [-1,1]
        """
        user_vec = np.array(embedding_user)
        game_vec = np.array(embedding_game)

        if np.linalg.norm(user_vec) == 0 or np.linalg.norm(game_vec) == 0:
            return 0.0

        similarity = np.dot(user_vec, game_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(game_vec))
        return float(similarity)

    @staticmethod
    def absa_similarity(user_aspects, game_aspects):
        """
        Compute ABSA similarity between user and game aspects.
        expected data format (MAY CHANGE LATER)
        Inputs:
            user_aspects: dict {aspect_name: score}
            game_aspects: dict {aspect_name: score}
        Output:
            similarity float in [0,1]
        """
        common_aspects = set(user_aspects.keys()).intersection(game_aspects.keys())
        if not common_aspects:
            return 0.0  # no common aspects

        user_vec = np.array([user_aspects[a] for a in common_aspects])
        game_vec = np.array([game_aspects[a] for a in common_aspects])

        # normalize
        if np.linalg.norm(user_vec) == 0 or np.linalg.norm(game_vec) == 0:
            return 0.0

        similarity = np.dot(user_vec, game_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(game_vec))
        return float(similarity)

    def compute_score(self, embedding_user, embedding_game, user_aspects, game_aspects, w_embedding=0.5, w_aspects=0.5):
        """
        Compute final score as weighted sum of embedding and ABSA similarities.
        """
        cos_sim = self.cosine_similarity(embedding_user, embedding_game)
        absa_sim = self.absa_similarity(user_aspects, game_aspects)
        return w_embedding * cos_sim + w_aspects * absa_sim


# if __name__ == "__main__":
#     scorer = GameScorer()

#     embedding_user = [0.1, 0.2, 0.3]
#     embedding_game = [0.2, 0.1, 0.3]

#     user_aspects = {"story": 1.0, "graphics": 0.5}
#     game_aspects = {"story": 0.8, "graphics": 0.9}

#     score = scorer.compute_score(embedding_user, embedding_game, user_aspects, game_aspects)
#     print("Final score:", score)
