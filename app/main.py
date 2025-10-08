import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
import re

# ---------- Utility functions ----------

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF."""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, n_words=350):
    """Split text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+n_words]) for i in range(0, len(words), n_words)]


def build_prompt(query, contexts):
    """Combine retrieved contexts and question into a single prompt."""
    context_text = "\n\n".join(contexts)
    prompt = (
        f"Answer the question based only on the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\nAnswer briefly:"
    )
    return prompt


# ---------- Load models once ----------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-small")
    return embed_model, qa_model


embed_model, qa_model = load_models()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="LLM Knowledge Miner", page_icon="🧠", layout="wide")
st.title("🧠 LLM Knowledge Miner with PDF Upload")
st.caption("Upload a PDF, and ask questions about its content locally.")

# Upload Section
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and processing text..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)
        df = pd.DataFrame({"text": chunks})
        st.success(f"✅ Loaded {len(chunks)} text chunks from your PDF.")

        # Create temporary embeddings and FAISS index
        embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Ask question
        query = st.text_input("🔍 Ask a question about your document:")
        k = st.slider("Top-k results", 1, 10, 3)

        if st.button("Search") and query.strip():
            q_emb = embed_model.encode([query])
            D, I = index.search(np.array(q_emb).astype("float32"), k)
            contexts = df.iloc[I[0]]["text"].tolist()

            with st.spinner("Generating answer..."):
                prompt = build_prompt(query, contexts)
                answer = qa_model(prompt, max_new_tokens=200)[0]["generated_text"]

            st.subheader("💬 Answer")
            st.write(answer)

            st.subheader("📚 Supporting Contexts")
            for i, c in enumerate(contexts, 1):
                with st.expander(f"Context {i}"):
                    st.write(c)
else:
    st.info("👆 Upload a PDF to begin.")
