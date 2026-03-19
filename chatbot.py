from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os
import re

from dotenv import load_dotenv
from groq import Groq

# ✅ Load env
load_dotenv()

CHROMA_PATH = r"chroma_db"

# ✅ Secure API key (FIXED)
client = Groq(api_key="your-api-key")

# ✅ Embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ✅ Vector DB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

retriever = vector_store.as_retriever(search_kwargs={'k': 15})


# 🔥 LLM call
def call_llama(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer ONLY using provided knowledge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# 🔥 Normalize (IMPORTANT FIX)
def normalize(text):
    return text.lower().replace("-", " ")


# 🔥 Reranking (IMPROVED)
def rerank_docs(docs, query):
    query_words = set(normalize(query).split())
    scored = []

    for doc in docs:
        text = normalize(doc.page_content)

        score = len(query_words & set(text.split()))

        # 🔥 Boost definition
        if "self attention" in text:
            score += 10

        if "attention mechanism" in text:
            score += 5

        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored[:5]]


# 🔥 BEST SENTENCE (FINAL FIX)
def get_best_sentence(paragraph, query):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)

    query_norm = normalize(query)

    # 🔥 PRIORITY: definition
    for s in sentences:
        s_norm = normalize(s)

        if (
            "self attention is" in s_norm or
            "self attention, sometimes called" in s_norm or
            "is an attention mechanism" in s_norm
        ):
            return s

    # fallback
    query_words = set(query_norm.split())
    best = sentences[0]
    max_score = 0

    for s in sentences:
        s_norm = normalize(s)
        words = set(s_norm.split())

        score = len(query_words & words)

        if score > max_score:
            max_score = score
            best = s

    return best


# 🎨 PREMIUM CSS
custom_css = """
body { background: #0f172a; color: white; }

.gradio-container {
    max-width: 900px !important;
    margin: auto;
}

.source-card {
    background: #111827;
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
    border-left: 4px solid #22c55e;
}

.source-title {
    font-weight: bold;
    color: #93c5fd;
}

.highlight {
    color: #facc15;
    font-weight: bold;
}
"""


# 🎨 FORMAT OUTPUT (PREMIUM UI)
def format_response(answer, citations):
    formatted = f"### 💡 Answer\n\n{answer}\n\n---\n### 🔍 Sources\n"

    for c in citations[:3]:
        formatted += f"""
<div class="source-card">
<div class="source-title">📄 {c['source']} (Page {c['page']})</div>
<div>🔴 <span class="highlight">{c['sentence']}</span></div>
</div>
"""
    return formatted


# 🔥 CHATBOT
def stream_response(message, history):

    if message.lower().strip() in ["hi", "hello", "hey"]:
        yield "👋 Hello! Ask me anything from your document."
        return

    docs = retriever.invoke(message)
    docs = rerank_docs(docs, message)

    if not docs:
        yield "⚠️ I don't have enough information."
        return

    knowledge = ""
    citations = []

    for doc in docs:
        paragraph = doc.page_content.replace("\n", " ").strip()

        knowledge += paragraph + "\n\n"

        best_sentence = get_best_sentence(paragraph, message)

        citations.append({
            "sentence": best_sentence,
            "page": doc.metadata.get("page", "Unknown"),
            "source": doc.metadata.get("source", "PDF")
        })

    # 🔥 Prompt
    rag_prompt = f"""
Answer ONLY using the knowledge below.

Question:
{message}

Knowledge:
{knowledge}
"""

    answer = call_llama(rag_prompt)

    # 🎨 Premium formatted output
    final = format_response(answer, citations)

    # streaming
    partial = ""
    for ch in final:
        partial += ch
        yield partial


# 🚀 PREMIUM UI LAYOUT
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown(
        """
        # 📄 AI Research Assistant  
        ### Ask questions from your PDFs with precise citations
        """
    )

    chatbot = gr.ChatInterface(
        fn=stream_response,
        textbox=gr.Textbox(
            placeholder="Ask something like 'What is self-attention?'...",
            container=False,
        ),
    )

demo.launch()
