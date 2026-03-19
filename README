# 🤖 Multimodal RAG Document Assistant

A next-generation RAG (Retrieval-Augmented Generation) chatbot capable of understanding **Text, Tables, and Charts** within PDF documents. 

Unlike standard chatbots that only read text, this system uses **Computer Vision** to "see" and interpret graphs, ensuring no data is left behind.

## 🚀 Key Features
* **Multimodal Ingestion:** Extracts and understands charts/figures using Google Gemini Vision.
* **Smart Table Parsing:** Preserves table structure using LlamaParse (Markdown mode) for accurate data retrieval.
* **Hybrid Search:** Combines Semantic Search (FAISS) with Keyword Search (BM25) to find specific metrics like "2.5%" or "2025".
* **Fail-Safe Architecture:** Includes rate-limiting and error handling to manage API quotas gracefully.

## 🛠️ Tech Stack
* **LLM:** Llama-3 (via Groq) for reasoning & answering.
* **Vision Model:** gemini-2.5-flash (via the newest `google-genai` SDK, forcing Developer API to avoid Vertex AI auth errors).
* **Orchestration:** LangChain (Core & Classic components).
* **Vector DB:** FAISS + BM25 (Ensemble Retriever).
* **Parsing:** LlamaParse (Text/Tables) + PyMuPDF (Images).
* **Backend:** FastAPI.

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables**
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key
    LLAMA_CLOUD_API_KEY=your_llama_parse_key
    ```

4.  **Run the Server**
    ```bash
    # Note: Clears previous credentials to avoid auth conflicts
    uvicorn main:app --reload
    ```

5.  **Use the API**
    * Open `http://127.0.0.1:8000`.
    * **Step 1:** Upload a PDF via `/upload-pdf`.
    * **Step 2:** Ask questions via `/ask` (e.g., *"What does Figure 8 show about LNG supply?"*).