# streamlit_app.py
import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from typing import List

load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Ensure a valid Windows-safe absolute path for Chroma
CHROMA_PERSIST_DIR = os.path.abspath(
    os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))
)

GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Helpers ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def pdf_to_text_bytes(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

@st.cache_resource(show_spinner=False)
def chroma_client(persist_directory: str):
    os.makedirs(persist_directory, exist_ok=True)  # Ensure folder exists
    return chromadb.PersistentClient(path=persist_directory)

def create_or_get_collection(client, name="docs"):
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)

def upsert_chunks(collection, texts: List[str], metadatas: List[dict], ids: List[str], embeddings):
    collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)

def embed_texts(model, texts: List[str]):
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def query_chroma(collection, query_embedding, top_k=4):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # removed 'ids' from include
    )
    ids = results["ids"][0]  # still available without including
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return list(zip(ids, docs, metas, dists))

def call_groq_chat(messages, model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=512):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

# --- Streamlit UI ---
st.set_page_config(page_title="studybot", layout="wide")
st.title("chat bot")

# Sidebar
st.sidebar.header("Settings")
persist_dir = st.sidebar.text_input("Chroma persist dir", CHROMA_PERSIST_DIR)
model_name = st.sidebar.text_input("LLM model (Groq)", "llama-3.3-70b-versatile")
top_k = st.sidebar.number_input("Top K documents", value=4, min_value=1, max_value=10)

client = chroma_client(persist_dir)
collection = create_or_get_collection(client, name="study_docs")
embed_model = get_embedding_model()

# Upload area
st.header("Upload documents (PDF / txt)")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "txt"])

if uploaded_files:
    st.info(f"Found {len(uploaded_files)} file(s). Processing...")
    all_chunks, all_metas, all_ids = [], [], []
    for f in uploaded_files:
        content = f.read()
        if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
            text = pdf_to_text_bytes(content)
        else:
            try:
                text = content.decode("utf-8")
            except:
                text = str(content)
        chunks = chunk_text(text)
        basename = os.path.basename(f.name)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metas.append({"source": basename, "chunk_index": i})
            all_ids.append(f"{basename}-{i}")
    if all_chunks:
        with st.spinner("Embedding chunks and upserting into Chroma..."):
            embeddings = embed_texts(embed_model, all_chunks)
            upsert_chunks(collection, all_chunks, all_metas, all_ids, embeddings)
        st.success(f"Upserted {len(all_chunks)} chunks into Chroma collection 'study_docs'.")

st.markdown("---")
st.header("Ask a question")

question = st.text_input("Your question", value="")
if st.button("Search & Generate") and question.strip():
    with st.spinner("Embedding query and searching Chroma..."):
        q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
        hits = query_chroma(collection, q_emb, top_k=top_k)

    context_parts = [
        f"Source: {meta.get('source')} | Chunk: {meta.get('chunk_index')}\n{doc_text}"
        for idx, doc_text, meta, dist in hits
    ]
    context_combined = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful, concise teaching assistant. Use only the provided CONTEXT to answer the user. "
        "If the answer is not contained in the context, say you don't know and suggest where to look in the sources."
    )
    user_prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context_combined}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    with st.spinner("Calling Groq LLM..."):
        try:
            resp_json = call_groq_chat(messages, model=model_name, temperature=0.0, max_tokens=512)
            choice = resp_json.get("choices", [{}])[0]
            message = choice.get("message", {}).get("content") or choice.get("text") or str(resp_json)
            st.subheader("Answer")
            st.write(message)

            st.subheader("Retrieved Sources (top-k)")
            for i, (idx, doc, meta, dist) in enumerate(hits):
                st.markdown(f"**{i+1}. {meta.get('source')} (chunk {meta.get('chunk_index')})** — distance: `{dist}`")
                st.write(doc[:1000] + ("..." if len(doc) > 1000 else ""))
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.json({"error": str(e), "response": getattr(e, 'response', None)})

st.markdown("""
---
**Notes**
- Uses Chroma's new `PersistentClient` API and updated `.query()` parameters.
- Automatically normalizes the Chroma path for Windows.
- Groq API endpoint is OpenAI-compatible. Ensure `GROQ_API_KEY` is set.
""")
