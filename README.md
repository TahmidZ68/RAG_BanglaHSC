# 🧠 Multilingual RAG System – Bangla & English (AI Engineer Level-1 Assessment)

This project implements a simple **Retrieval-Augmented Generation (RAG)** pipeline that supports both **Bangla and English** queries. It retrieves grounded answers from the **HSC Bangla 1st Paper textbook** using semantic search and generates multilingual answers using a transformer-based language model.

---

## ✅ Features

- 🔍 **Semantic search** over Bangla textbook content (PDF with OCR)
- 💬 Accepts queries in **Bangla and English**
- 🤖 Uses `facebook/mbart-large-50-many-to-many-mmt` for multilingual answer generation
- 🧠 Maintains both **short-term (chat)** and **long-term (PDF vector)** memory
- 🌐 Lightweight **REST API** using FastAPI

---

## 🧩 Tools & Libraries Used

| Component           | Library / Tool                                   |
|--------------------|--------------------------------------------------|
| PDF Text Extraction | `pdf2image`, `pytesseract`, `pdfplumber`         |
| OCR Language        | `ben` (Bangla-trained Tesseract OCR)             |
| Chunking Strategy   | Sentence-based using Bangla delimiter (।)        |
| Embedding Model     | `distiluse-base-multilingual-cased-v1` (SBERT)   |
| Vector Store        | `FAISS`                                          |
| Language Model      | `facebook/mbart-large-50-many-to-many-mmt`       |
| REST API Framework  | `FastAPI`, `Uvicorn`                             |

---

## 🛠️ Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bangla-rag-system.git
cd bangla-rag-system
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
You may also need OCR & PDF dependencies:
bash
Copy
Edit
sudo apt install tesseract-ocr tesseract-ocr-ben poppler-utils
3. Run the FastAPI App
bash
Copy
Edit
uvicorn rag_api:app --reload --port 8000
Visit: http://127.0.0.1:8000/docs for Swagger UI

🚀 Sample API Usage
POST /query

Request:
json
Copy
Edit
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
Response:
json
Copy
Edit
{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শম্ভুনাথ",
  "context": [
    "অনুপমের ভাষায়, শম্ভুনাথ ছিল একজন সুপুরুষ।",
    "শম্ভুনাথ ছিল অনুপমের মামা।"
  ]
}
📈 Evaluation Matrix
Metric	Implementation
Groundedness	Answers are generated only from retrieved context
Relevance	Top-K semantic cosine similarity using FAISS
Manual QA	Verified with real HSC questions in Bangla

🧪 Sample Queries & Outputs
Query (Bangla)	Expected Answer
অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?	শম্ভুনাথ
কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?	মামা
বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?	১৫ বছর

💡 Assessment Questions – Answered
1. What method or library did you use to extract the text, and why?
I used a hybrid of pdf2image and pytesseract (with Bengali OCR - ben) for extracting text from the HSC textbook, which is scanned and image-heavy. Additionally, pdfplumber was used for text-based PDFs (fallback). OCR was essential to overcome the font encoding challenges typical in Bangla texts.

Challenge: Inconsistent line breaks and Bangla character splits.

Solution: Custom preprocessing pipeline – line joining, punctuation normalization, and Bangla-specific sentence segmentation.

2. What chunking strategy did you choose, and why?
I used sentence-based chunking using Bangla punctuation delimiters (।). This preserves semantic units, which improves retrieval accuracy while preventing context mixing.

For example: Chunking by paragraphs often diluted direct Q&A-style pairs. Sentence-level chunks gave more precise matches.

3. What embedding model did you use, and why?
Used distiluse-base-multilingual-cased-v1 from Sentence Transformers, which supports 100+ languages, including Bangla and English.

It gives dense semantic representations, ideal for cross-lingual similarity.

Better performance than traditional TF-IDF or Bag-of-Words on Bangla texts.

4. How are you comparing the query with your stored chunks?
I used cosine similarity with FAISS to compare the query embedding with vectorized chunks. FAISS offers fast similarity search, even with large corpora, and supports future scaling.

Why Cosine? Because it compares semantic direction, not magnitude – useful when embeddings are normalized.

5. How do you ensure meaningful comparisons?
Each chunk is normalized, cleaned, and semantically consistent.

Queries are embedded using the same SBERT model to maintain encoding consistency.

If a query is vague or lacks context:

The system fetches semantically closest matches.

Generic or ambiguous answers may appear — future improvements include integrating clarification prompts or top-3 chunk re-ranking using cross-encoders.

6. Do the results seem relevant? How could it improve?
Yes — the answers for factual questions are accurate and grounded.

However, improvements can be made via:

🧩 Smarter chunking: overlapping or hybrid sentence-paragraphs

🧠 Instruction-tuned models: e.g., Flan-T5-multilingual

🧪 Cross-encoder reranking: instead of only dense retrieval

🔁 Reinforcement learning from user feedback

⚙️ Project Structure
bash
Copy
Edit
.
├── rag_api.py                 # FastAPI application (API endpoints)
├── preprocess.py              # PDF processing and chunking
├── vector_store.py            # FAISS creation & similarity search
├── Assessment_Tahmid.ipynb    # Development and testing notebook
├── requirements.txt           # Python dependencies
├── README.md
⚠️ Note on OpenAI Usage
I attempted to integrate gpt-3.5-turbo for generation and refinement. However, billing limitations on my OpenAI account prevented successful API calls. As a fallback, I used the open-source multilingual model facebook/mbart-large-50-many-to-many-mmt.

For production, OpenAI or Mistral models with function calling can further enhance accuracy and instruction alignment.

✍️ Author
Name: Md Tahmid Zoayed
Email: tahmidzoayed@gmail.com
Submitted For: AI Engineer (Level-1) Technical Assessment

🔮 Future Improvements
🚀 Add PDF uploader for dynamic corpus building

💬 Implement chat memory for short-term conversational context

📦 Dockerize the application for deployment

🧠 Integrate Bangla-tuned Flan-T5 or custom finetuned Bengali models

⚖️ Add weighted retrieval filtering based on cosine + keyword scoring

📘 References
FAISS by Facebook AI

Multilingual SBERT

MBART Multilingual Model

Tesseract OCR with Bengali
