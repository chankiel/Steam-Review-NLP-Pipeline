# PineconeManager.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from SentenceEmbedder import SentenceEmbedder

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")
DIMENSION = int(os.getenv("DIMENSION"))
API_KEY = os.getenv("PINECONE_API_KEY")
ENV = os.getenv("PINECONE_ENV")

class PineconeManager:
    def __init__(self):
        # Pinecone client
        self.pc = Pinecone(api_key=API_KEY)
        self.dimension = DIMENSION
        self.index_name = INDEX_NAME

        # create index if not exists
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=ENV)
            )

        self.index = self.pc.Index(self.index_name)

        # initialize sentence embedder
        self.embedder = SentenceEmbedder()

    # --- New cleaner method ---
    def upsert_review(self, id_, review_text, app_id, app_name):
        vec = self.embedder.embed_sentence(review_text)

        metadata = {
            "app_id": app_id,
            "app_name": app_name,
            "review_text": review_text
        }

        self.index.upsert(vectors=[
            {
                "id": id_,
                "values": vec.tolist(),
                "metadata": metadata
            }
        ])

    # --- Query ---
    def query_text(self, text, top_k=5):
        vec = self.embedder.embed_sentence(text)
        res = self.index.query(
            vector=vec.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return res["matches"]


# --- Main trial ---
# if __name__ == "__main__":
#     pm = PineconeManager()

#     # Example review info
#     review_id = "rev_001"
#     app_id = "com.riot.league"
#     app_name = "League of Legends"
#     review_text = "I currently play league of legends and I love it"

#     # Trial embedding shape
#     sample_vec = pm.embedder.embed_sentence(review_text)
#     print("Trial embedding vector shape:", sample_vec.shape)

#     # Upsert trial
#     pm.upsert_review(
#         id_=review_id,
#         review_text=review_text,
#         app_id=app_id,
#         app_name=app_name
#     )
#     print("Trial upsert done.")

#     # Query test
#     print("\nQuery: 'I LOVE league of legends'")
#     results = pm.query_text("I LOVE league of legends")
#     for r in results:
#         print(r)

#     print("\nQuery: 'I hate league of legends'")
#     results = pm.query_text("I hate league of legends")
#     for r in results:
#         print(r)
