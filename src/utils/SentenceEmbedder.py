from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceEmbedder:
    """
    Class to convert sentences (reviews, comments, or game descriptions) into embeddings
    using Sentence-Transformers models.
    """
    def __init__(self, model_name="microsoft/mpnet-base"):
        """
        model_name: HF / SentenceTransformers model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_sentence(self, sentence):
        """
        Embed a single sentence
        Returns: numpy array
        """
        embedding = self.model.encode(sentence, convert_to_numpy=True)
        return embedding

    def embed_sentences(self, sentences):
        """
        Embed multiple sentences at once
        sentences: list of strings
        Returns: numpy array of shape (len(sentences), embedding_dim)
        """
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        return embeddings

# if __name__ == "__main__":
#     embedder = SentenceEmbedder()

#     # Single sentence
#     sent = "I love the story and graphics of this game!"
#     vec = embedder.embed_sentence(sent)
#     print("Embedding shape:", vec.shape)

#     # Multiple sentences
#     sentences = [
#         "The story is amazing.",
#         "Graphics could be better.",
#         "Multiplayer is fun!"
#     ]
#     vecs = embedder.embed_sentences(sentences)
#     print("Embeddings shape:", vecs.shape)
