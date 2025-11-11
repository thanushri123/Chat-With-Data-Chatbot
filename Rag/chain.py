# rag/chain.py
"""
LangChain retrieval-augmented chain:
1. Embed query → FAISS → top-k rows
2. Prompt LLM for a short answer **and** optional chart JSON
3. Return a Python dict ready for the Flask layer.
"""

import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.schema import StrOutputParser

# -------------------------------------------------
# 1. Load FAISS + metadata (lightweight wrapper)
# -------------------------------------------------
def load_faiss_store():
    embedder = SentenceTransformer("thenlper/gte-small")
    index = faiss.read_index("rag/faiss.index")
    with open("rag/meta.pkl", "rb") as f:
        metas = pickle.load(f)

    class SimpleFAISS:
        def __init__(self, emb, idx, meta):
            self.embedder = emb
            self.index = idx
            self.meta = meta

        def similarity_search(self, query, k=8):
            qvec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
            faiss.normalize_L2(qvec)
            _, I = self.index.search(qvec, k)
            return [self.meta[i] for i in I[0]]

    return SimpleFAISS(embedder, index, metas)

vector_store = load_faiss_store()

# -------------------------------------------------
# 2. LLM (swap with Ollama if you don’t want OpenAI)
# -------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------------------------------
# 3. Prompt – forces JSON output
# -------------------------------------------------
PROMPT_TEMPLATE = """You are a friendly data analyst.
Answer in **two** parts:

1. A short natural-language answer (max 2 sentences).
2. If a chart makes sense, a JSON object: {{"chart": {{"type": "...", "x": "...", "y": "...", "hue": "..."}}}}

If no chart is needed, just return {{"answer": "..."}}.

Context rows (JSON):
{context}

User question: {question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# -------------------------------------------------
# 4. Retrieval helper
# -------------------------------------------------
retriever = RunnableLambda(lambda q: vector_store.similarity_search(q, k=8))

def format_context(docs):
    return "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)

# -------------------------------------------------
# 5. Full chain
# -------------------------------------------------
chain = (
    {"question": lambda x: x, "context": retriever | format_context}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------
# 6. Public function used by nlp_pipeline
# -------------------------------------------------
def rag_answer(user_query: str) -> dict:
    raw = chain.invoke(user_query).strip()
    # Clean possible markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"answer": raw}
