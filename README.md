# 🚀 Tugas Besar IF5153 NLP — Steam Review Pipeline

## 📌 Daftar Isi
- [Deskripsi](#deskripsi)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Inference](#inference)
- [Struktur Proyek](#struktur-proyek)
- [Kelompok](#kelompok)

---

## 📝 Deskripsi <a name="deskripsi"></a>

**Steam Review Pipeline** adalah sistem lengkap untuk memproses, menganalisis, dan mengekstrak insight dari ulasan game di Steam.  
Pipeline ini terdiri dari **tiga model utama**:

1. **ABSA (Aspect-Based Sentiment Analysis)**  
   Mengklasifikasikan sentimen per aspek (gameplay, story, audio, graphic, dsb).

2. **Summarizer (NLG)**  
   Menghasilkan ringkasan dari kumpulan review sebuah game.

3. **Recommender System**  
   Menggunakan **Vector DB** untuk menemukan game mirip berdasarkan embedding ulasan, kemudian merangkum hasil rekomendasi menggunakan LLM.

Pipeline mendukung inference melalui **CLI REPL** dan **notebook**.

---

## 📦 Dependencies <a name="dependencies"></a>

- Python ≥ 3.10  
- PyTorch  
- Transformers  
- SentenceTransformers  
- Pandas  
- tqdm  
- ChromaDB / FAISS (Vector DB)  
- Jupyter Notebook

Install semua dependencies:

```sh
pip install -r requirements.txt
```

## ⚙️ Setup <a name="setup"></a>
1. Clone repository    
    ```sh
    git clone <your_repo_url>
    cd steam-review-pipeline
    ```

2. Buat virtual environment (venv)  
    ```sh
    python -m venv venv
    ```

3. Aktifkan venv  
    Windows: `venv\Scripts\activate`  
    macOS / Linux: `source venv/bin/activate`

4. Install dependencies
```sh
pip install -r requirements.txt
```

## 🔮 Inference <a name="inference"></a>
1. **ABSA (Aspect-Based Sentiment Analysis)**  
    ABSA dapat dijalankan melalui CLI REPL:
    ```sh
    python -m src.inference.ABSA.main
    ```  
    Contoh:
    ```sh
    ABSA CLI ready.
    Review> the gameplay is fun but the graphics are terrible  

    Output:
    gameplay      → positive
    graphics      → negative
    story         → neutral
    ...
    ```  

2. **Summarizer**
    Summarizer lengkap (batching / per game) dapat dijalankan melalui notebook

3. **Recommender System (NLG)**  
    Pipeline rekomendasi menggunakan Vector DB dan LLM summarization.

    Jalankan:
    ```sh
    python -m src.inference.NLG.main
    ```  
    Output meliputi:  
    - Daftar game mirip
    - Penjelasan kenapa game tersebut cocok
    - Ringkasan rekomendasi oleh LLM

## 📂 Struktur Proyek <a name="struktur-proyek"></a>
```
project/
│
├── models/
│    ├── checkpoints/
│    └── best/
│
├── src/
│   ├── inference/
│   │   ├── ABSA/
│   │   └── NLG/
│   │
│   ├── training/
│   │   ├── datasets/
│   │   ├── evaluator/
│   │   ├── models/
│   │   └── trainers/
│   │
│   ├── utils/
│   └── preprocess/
│
├── notebooks/
│   ├── ABSA/
│   ├── common/
│   ├── Summarizer/
│   └── NLG/
│
├── configs/
├── requirements.txt
└── README.md
```  

## 👥 Kelompok 10 — dua_ganteng_satu_cadel <a name="kelompok"></a>
- Ignatius Jhon Hezkiel Chan - 13522029
- Daniel Mulia Putra Manurung - 13522043
- Matthew Vladimir Hutabarat - 13522093  