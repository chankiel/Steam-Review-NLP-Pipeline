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


# # --- Main: Full Pipeline (Clear Sections) ---
# if __name__ == "__main__":
#     import pandas as pd

#     pm = PineconeManager()

#     # ============================================================
#     # (A) CLEAR INDEX
#     # ============================================================
#     print("\n================= CLEARING INDEX =================")
#     try:
#         pm.index.delete(delete_all=True)
#         print("[INFO] Index cleared successfully.")
#     except Exception as e:
#         print("[ERROR] Failed clearing index:", e)

#     # ============================================================
#     # (B) LOAD CSV
#     # ============================================================
#     print("\n================= LOADING CSV =================")
#     df = pd.read_csv("data2.csv")

#     print(f"[INFO] Loaded {len(df)} rows from data2.csv")
#     print(df.head())

#     # ============================================================
#     # (C) INSERT EACH ROW INTO PINECONE
#     # ============================================================
#     print("\n================= UPSERTING REVIEWS =================")

#     for i, row in df.iterrows():
#         review_id = f"rev_{row['app_id']}"
#         app_id = row["app_id"]
#         app_name = row["app_name"]
#         review_text = row["summary"]

#         print(f"\n--- Upserting Review {review_id} ---")
#         print("Game:", app_name)
#         print("Review:", review_text[:80], "...")

#         pm.upsert_review(
#             id_=review_id,
#             review_text=review_text,
#             app_id=str(app_id),
#             app_name=app_name
#         )

#     print("\n[INFO] All rows inserted.")

#     # ============================================================
#     # (D) QUERY TEST
#     # ============================================================
#     print("\n================= QUERY TEST =================")

#     test_queries = [
#         "fast-paced shooting game",
#         "I love the graphics but hate grenade mechanics",
#         "classic competitive fps",
#     ]

#     for q in test_queries:
#         print(f"\n>>> QUERY: {q}")
#         results = pm.query_text(q)

#         for r in results:
#             print(r)

#     print("\n================= DONE =================")

