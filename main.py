import os
import shutil

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from google import genai
from google.genai import types
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import threading

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from llama_parse import LlamaParse
import fitz
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(vertexai=False, api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_FAISS_PATH = "vectorstore/db_faiss_v2"
conversation_history = []
last_interaction_time = time.time()
TIMEOUT_SECONDS = 3600  # 1 hour

# --- Globals cached at startup (never re-created per query) ---
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model ready.")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

pdf_memory = None       # FAISS index
bm25_retriever = None   # BM25 index


def update_interaction():
    global last_interaction_time
    last_interaction_time = time.time()


def cleanup_loop():
    """Background thread: deletes vectorstore after 1 hour of inactivity."""
    global pdf_memory, bm25_retriever, conversation_history, last_interaction_time
    while True:
        time.sleep(60)
        elapsed = time.time() - last_interaction_time
        if elapsed > TIMEOUT_SECONDS and os.path.exists(DB_FAISS_PATH):
            print(f"Inactivity detected ({elapsed:.0f}s). Cleaning up...")
            try:
                shutil.rmtree(DB_FAISS_PATH)
                pdf_memory = None
                bm25_retriever = None
                conversation_history = []
                print("Cleanup complete.")
            except Exception as e:
                print(f"Cleanup error: {e}")


cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
cleanup_thread.start()


def summarize_image(image_bytes):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                """Analyze this image in detail and describe ALL of the following:

1. LAYOUT & STRUCTURE: Describe the spatial arrangement — what is at the center, what surrounds it, how elements are positioned relative to each other (top, bottom, left, right, inside, outside, overlapping).

2. If it is a DIAGRAM (e.g. lifecycle, model, framework, flowchart):
   - List EVERY labeled element with its exact position in the diagram.
   - Explicitly state what element is at the CENTER of the diagram (if any).
   - Describe what is at the top, bottom, left, right, and middle.
   - Describe the overall shape or structure (circle, triangle, arrow, grid, etc.).

3. If it is a CHART or GRAPH:
   - Extract specific numbers, percentages, and values shown.
   - Describe trends, axis labels, legend meanings, and data series.

4. If it is a TABLE:
   - Summarize key rows and columns with their values.
   - Highlight important comparisons between columns (e.g. different years).

5. CONNECTIONS & ARROWS:
   - Describe every arrow, line, or connector — what it links and its direction.
   - Note whether arrows are single-headed, double-headed, curved, or straight.

6. TEXT LABELS:
   - List all visible text labels exactly as written in the image.

Be extremely specific about positions and relationships. Do not omit any labeled element."""
            ]
        )
        return response.text
    except Exception as e:
        print("Gemini Vision Error:", e)
        return "[Image caption failed]"


# Load existing vectorstore and build BM25 cache at startup
if os.path.exists(DB_FAISS_PATH):
    print("Found existing vectorstore. Loading...")
    try:
        pdf_memory = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        docstore_docs = list(pdf_memory.docstore._dict.values())
        if docstore_docs:
            bm25_retriever = BM25Retriever.from_documents(docstore_docs)
            bm25_retriever.k = 6
        print("Vectorstore and BM25 index ready.")
    except Exception as e:
        print(f"Failed to load existing vectorstore: {e}")


def process_multimodal_pdf(pdf_path: str):
    global pdf_memory, bm25_retriever  # must declare global or assignments stay local
    print("--- STARTING INGESTION ---")

    try:
        parser = LlamaParse(
            result_type="markdown",
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            verbose=True
        )
        parsed_docs = parser.load_data(pdf_path)
    except Exception as e:
        print("LlamaParse Error:", e)
        return 0

    doc = fitz.open(pdf_path)
    full_combined_text = ""

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        images = page.get_images(full=True)
        drawings = page.get_drawings()
        has_visuals = len(images) > 0 or len(drawings) > 0

        caption = ""
        if has_visuals:
            print(f"Visuals found on Page {page_num}. Rendering page...")
            try:
                time.sleep(2.5)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                caption = summarize_image(img_bytes)

                if "429" in caption or "quota" in caption.lower():
                    print(f"Skipping Page {page_num} due to Rate Limit.")
                    caption = ""
                else:
                    caption = f"\n\n[VISUAL ANALYSIS OF PAGE {page_num}]\n{caption}\n[END VISUAL ANALYSIS]\n"
            except Exception as e:
                print(f"Visual Analysis Failed for Page {page_num}: {e}")
                caption = ""

        text_content = ""
        if page_index < len(parsed_docs):
            text_content = parsed_docs[page_index].text

        full_combined_text += f"--- PAGE {page_num} ---\n{text_content}\n{caption}\n\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400)
    chunks = splitter.split_text(full_combined_text)
    documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]

    pdf_memory = FAISS.from_documents(documents, embeddings)
    pdf_memory.save_local(DB_FAISS_PATH)

    # Rebuild BM25 cache after new upload
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6

    print(f"--- INGESTION COMPLETE: {len(chunks)} chunks stored. ---")
    return len(chunks)


def get_ai_response(query: str):
    global conversation_history

    if pdf_memory is None:
        return "System Error: Please re-upload the PDF to initialize the database."

    faiss_retriever = pdf_memory.as_retriever(search_kwargs={"k": 6})

    if bm25_retriever:
        retriever_to_use = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
    else:
        retriever_to_use = faiss_retriever

    clean_history = []
    for msg in conversation_history[-5:]:
        role = msg['role']
        content = msg['content'].replace("{", "(").replace("}", ")")
        clean_history.append(f"{role}: {content}")
    history_str = "\n".join(clean_history)

    prompt_template_str = f"""
    You are an expert Business and Financial Analyst.

    # CHAT HISTORY:
    {history_str}

    # CONTEXT:
    {{context}}

    # USER QUESTION:
    {{input}}

    # INSTRUCTIONS:
    0. Answer from the provided context. If the topic is partially covered, share what IS available and note it may be incomplete. Only refuse if the topic is completely absent.

    1. Figures & Diagrams: When the question refers to a figure, diagram, model, or visual element, give PRIORITY to the [VISUAL ANALYSIS OF PAGE X] sections. These contain the definitive spatial description.

    2. Spatial / Position Questions: If asked about the centre, middle, top, bottom, left, or right of a diagram, answer STRICTLY from the [VISUAL ANALYSIS] block. Do NOT infer from surrounding text.

    3. Figure Identification: If asked about a specific figure number, find the matching [VISUAL ANALYSIS] block and describe all labeled elements, positions, arrows, and connections.

    4. Charts & Graphs: Use [VISUAL ANALYSIS] sections for chart values, trends, axis labels, and legend meanings.

    5. Exact Numbers: Use EXACT numbers from the context. Do not approximate.

    6. Tables: Read column headers carefully. Do not confuse years or column values.

    Answer:
    """

    prompt = PromptTemplate(input_variables=["context", "input"], template=prompt_template_str)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_to_use, document_chain)

    response = retrieval_chain.invoke({"input": query})
    answer = response["answer"]

    conversation_history.append({"role": "User", "content": query})
    conversation_history.append({"role": "AI", "content": answer})

    return answer


@app.post("/reset-session")
async def reset_session():
    """Clears vectorstore and history for a fresh upload."""
    global pdf_memory, bm25_retriever, conversation_history

    pdf_memory = None
    bm25_retriever = None
    conversation_history = []

    if os.path.exists(DB_FAISS_PATH):
        try:
            shutil.rmtree(DB_FAISS_PATH)
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": True}


@app.get("/check-session")
async def check_session():
    """Returns True if vectorstore exists, so UI can skip the upload step."""
    update_interaction()
    if os.path.exists(DB_FAISS_PATH):
        return {"ready": True}
    return {"ready": False}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    update_interaction()
    global conversation_history
    conversation_history = []

    if not file.filename.lower().endswith(".pdf"):
        return {"success": False, "error": "Invalid file type."}

    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        num_chunks = process_multimodal_pdf(temp_filename)
        return {"success": True, "chunks": num_chunks}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/ask")
async def ask(question: str = Form(...)):
    update_interaction()
    if pdf_memory is None and not os.path.exists(DB_FAISS_PATH):
        return {"success": False, "result": "Please upload a PDF first."}
    return {"success": True, "result": get_ai_response(question)}


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/home")
async def home():
    return {"message": "Multi-Modal RAG API is Live."}