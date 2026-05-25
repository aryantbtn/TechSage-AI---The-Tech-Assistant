import os # Step 4
import re
import json # Step 1
import hashlib # Step 4.1
import streamlit as st # Step 6.4
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document # Step 1
from langchain_community.document_loaders import PyPDFLoader # Step 2
from langchain_text_splitters import RecursiveCharacterTextSplitter # Step 3
from langchain_community.embeddings import HuggingFaceEmbeddings # Step 6.2
from langchain_community.vectorstores import Chroma # Step 4
from transformers import pipeline # Step 6.3

import mysql.connector
import time
import math


@st.cache_resource
def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embedding_model

@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

@st.cache_resource
def load_llm():
    llm_pipeline = pipeline(
        task="text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        temperature=0.5,
        device_map="auto"
    )

    return llm_pipeline


# MySQL Server return function -> credentials
def get_connection():
    conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="Aryan@7105",
        database="rag_feedback",
        autocommit = True
    )

    conn.ping(reconnect=True)

    return conn

# MySQL Feedback Data Base
def init_feedback_db():

    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS response_feedback (

            id INT AUTO_INCREMENT PRIMARY KEY,

            query TEXT,

            chunk_hash VARCHAR(64),

            source_file TEXT,

            page_number VARCHAR(255),

            chunk LONGTEXT,

            generated_response LONGTEXT,

            reward INT,

            created_at BIGINT,

            INDEX idx_chunk_hash (chunk_hash),
            INDEX idx_created_at (created_at)
        )
    """)

    conn.commit()
    conn.close()


def save_individual_feedback(query, response, chunk, metadata, reward):
    conn = None

    try:
        conn = get_connection()

        c = conn.cursor()

        chunk = chunk.encode(
            "utf-8",
            errors="ignore"
        ).decode("utf-8")

        response = response.encode(
            "utf-8",
            errors="ignore"
        ).decode("utf-8")

        chunk_hash = hashlib.sha256(
            chunk.encode("utf-8")
        ).hexdigest()

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = str(
            metadata.get(
                "page",
                -1
            )
        )

        ts = int(time.time())

        print("INSERT VALUES:")
        print(query)
        print(chunk_hash)
        print(source)
        print(page)
        print(reward)

        c.execute("""
            INSERT INTO response_feedback
            (
                query,
                chunk_hash,
                source_file,
                page_number,
                chunk,
                generated_response,
                reward,
                created_at
            )

            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            query,
            chunk_hash,
            source,
            page,
            chunk,
            response,
            reward,
            ts
        ))

        conn.commit()

        print("Feedback inserted successfully.")

        c.execute("SELECT COUNT(*) FROM response_feedback")

        print(c.fetchone())

    except Exception as e:
        print("ERROR:", e)

        if conn:
            conn.rollback()

    finally:
        if conn and conn.is_connected():
            conn.close()

def get_feedback_data(chunk):

    conn = get_connection()

    c = conn.cursor()

    chunk_hash = hashlib.sha256(
        chunk.encode("utf-8")
    ).hexdigest()

    c.execute("""
        SELECT
            SUM(CASE WHEN reward = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN reward = -1 THEN 1 ELSE 0 END),
            MAX(created_at)

        FROM response_feedback

        WHERE chunk_hash = %s
    """, (chunk_hash,))

    result = c.fetchone()

    conn.close()

    if result:

        positive = result[0] or 0

        negative = result[1] or 0

        last_updated = result[2] or 0

        return positive, negative, last_updated

    return 0, 0, 0

def normalize_reranker(score):
    return 1 / (1 + math.exp(-score))

def normalize_feedback(positive, negative):
    return positive / (positive + negative + 1)

def normalize_recency(timestamp):
    if timestamp == 0:
        return 0
    age = time.time() - timestamp
    return math.exp(-age / 86400)  # decay over 1 day


# -----------------------------
# Step 1: Load JSON Data
# -----------------------------
def load_json_documents(path):

    with open(path, "r") as f:
        data = json.load(f)

    docs = []

    for idx, item in enumerate(data):

        docs.append(

            Document(

                page_content=
                f"Question: {item['question']}\n"
                f"Answer: {item['answer']}",

                metadata={

                    "source": path,

                    # JSON HAS NO PAGES
                    "page": f"JSON Entry {idx+1}",

                    "type": "json",

                    "chunk_id": idx,

                    "timestamp": time.time()
                }
            )
        )

    return docs


# -----------------------------
# Step 2: Load PDF Data
# -----------------------------
def load_pdf_documents(files):
    docs = []
    for file in files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    return docs


# -----------------------------
# Step 3: Chunking
# -----------------------------
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

# -----------------------------
# Step 4: Setup / Update Vector DB
# -----------------------------
def setup_or_update_chroma(chunks, embedding_model):
    if os.path.exists("./chroma_db"):
        print("Loading existing Chroma DB...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_model
        )

        if chunks:
            print("Updating documents...")

            unique_sources = set(
                chunk.metadata["source"]
                for chunk in chunks
            )

            for source in unique_sources:
                vectorstore.delete(
                    where={"source": source}
                )

            # ADD NEW CHUNKS
            vectorstore.add_documents(chunks)

            vectorstore.persist()

        return vectorstore

    print("Creating new Chroma DB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()

    return vectorstore

# -----------------------------
# Step 4.1: File Tracking
# -----------------------------
def load_file_tracker():
    if os.path.exists("file_tracker.json"):
        with open("file_tracker.json", "r") as f:
            return json.load(f)
    return {}

def save_file_tracker(tracker):
    with open("file_tracker.json", "w") as file:
        json.dump(tracker, file)

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# -----------------------------
# Step 4.2: Load ONLY new PDFs
# -----------------------------
def load_new_documents(files, tracker):
    new_docs = []

    for file in files:

        current_hash = get_file_hash(file)

        if file not in tracker or tracker[file] != current_hash:

            print(f"Processing new/updated file: {file}")

            loader = PyPDFLoader(file)

            docs = loader.load()

            # ADD CUSTOM METADATA
            for doc in docs:
                doc.metadata["source"] = file
                doc.metadata["timestamp"] = time.time()

            new_docs.extend(docs)

            tracker[file] = current_hash

    return new_docs, tracker


# -----------------------------
# Step 5: RAG Response
# -----------------------------
def get_rag_response(query, vectorstore, llm, embedding_model, reranker):
    docs = vectorstore.max_marginal_relevance_search(
        query,
        k=20,
        fetch_k=40
    )

    print(f"Retrieved docs before reranking: {len(docs)}")

    # BUILD QUERY-CHUNK PAIRS
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    # CROSS-ENCODER SCORES
    reranker_scores = reranker.predict(pairs)
    scored_docs = []

    for doc, reranker_score in zip(docs, reranker_scores):

        reranker_score = normalize_reranker(
            reranker_score
        )

        if reranker_score < 0.3:
            continue

        positive_votes, negative_votes, last_updated = \
            get_feedback_data(doc.page_content)

        feedback_score = normalize_feedback(
            positive_votes,
            negative_votes
        )

        recency_score = normalize_recency(
            last_updated
        )

        feedback_score = float(feedback_score or 0)
        recency_score = float(recency_score or 0)

        alpha = 0.6
        beta = 0.25
        gamma = 0.15

        final_score = (
                alpha * reranker_score +
                beta * feedback_score +
                gamma * recency_score
        )

        scored_docs.append(
            (
                final_score,
                reranker_score,
                feedback_score,
                recency_score,
                doc
            )
        )

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # DEBUG Step
    print("\n===== RERANK RESULTS =====")

    for item in scored_docs:
        print(
            f"""
                Final: {item[0]:.3f}
                Reranker: {item[1]:.3f}
                Feedback: {item[2]:.3f}
                Recency: {item[3]:.3f}
                """
        )

    top_docs = [
        item[4] for item in scored_docs[:5]
    ]

    individual_responses = []

    # GENERATE RESPONSE FOR EACH CHUNK
    for doc in top_docs:
        prompt = f"""
            You are an expert IT support assistant.

            Answer ONLY from the provided context.

            If answer is not found say:
            "I don't know based on this source."

            Context:
            {doc.page_content}

            User Question:
            {query}

            Give precise answer:
            """

        response = llm(
            prompt,
            max_new_tokens=200,
            temperature=0.3
        )[0]["generated_text"]

        final_response = response.split(
            "Give precise answer:"
        )[-1].strip()

        individual_responses.append({

            "response": final_response,

            "chunk": doc.page_content,

            "metadata": doc.metadata
        })
    return individual_responses


# -------------------------------------------------------------
# Step 6: Feedback Layer - Store the user feedback response
# -------------------------------------------------------------
def save_training_feedback(query, context, response, feedback):
    data = {
        "query": query,
        "context": context,
        "response": response,
        "reward": 1 if feedback == "👍 Yes" else 0
    }

    with open("feedback_log.json", "a") as f:
        f.write(json.dumps(data) + "\n")


# -----------------------------
# Step 7: Main Pipeline
# -----------------------------
def main():
    # Initialize database FIRST
    init_feedback_db()

    reranker = load_reranker()

    # Step 1.1 -> Load Json file data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, "..", "pdf", "json-question-for-model-training.json")
    json_docs = load_json_documents(json_path)

    # Step 7.1 -> PDF paths
    files = [
        os.path.join(BASE_DIR, "..", "pdf", "It-Support Textbook.pdf"),
        os.path.join(BASE_DIR, "..", "pdf", "IT-troubleshootGuide.pdf"),
        os.path.join(BASE_DIR, "..", "pdf", "IT-Support.pdf")
    ]

    # Step 7.2 -> Embedding model
    embedding_model = load_embedding_model()

    # Step 7.1.1 → Load tracker
    tracker = load_file_tracker()

    # Step 7.1.3 → Load ONLY new/updated PDFs
    new_pdf_docs, tracker = load_new_documents(files, tracker)

    # Step 7.1.4 → Save tracker
    save_file_tracker(tracker)

    # Step 7.1.5 → Decide what to process
    if not os.path.exists("./chroma_db"):
        all_docs = json_docs + new_pdf_docs
    else:
        all_docs = new_pdf_docs

    # Step 3.1 -> Chunking
    chunks = create_chunks(all_docs)

    # Step 4 -> Save chunks vector embedding into ChromaDB
    vectorstore = setup_or_update_chroma(chunks, embedding_model)

    # Step 7.3 -> LLM
    llm = load_llm()

    # Step 7.4 -> Streamlit UI
    st.title("IT Support Chatbot (RAG + LLaMA)")

    # Step 7.4.1 → Session State (Chat History)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_results" not in st.session_state:
        st.session_state.last_results = []

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    if "feedback_status" not in st.session_state:
        st.session_state.feedback_status = {}

    # Step 7.4.2 → Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # USER INPUT
    user_query = st.chat_input(
        "Ask your IT support question..."
    )

    # PROCESS NEW QUERY
    if user_query:
        st.session_state.last_query = user_query

        # SHOW USER MESSAGE
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        # GENERATE RAG RESPONSE
        results = get_rag_response(
            user_query,
            vectorstore,
            llm,
            embedding_model,
            reranker
        )

        st.session_state.last_results = results

    # DISPLAY RESPONSES
    if st.session_state.last_results:

        with st.chat_message("assistant"):

            for idx, item in enumerate(st.session_state.last_results, start=1):

                response = item["response"]

                chunk = item["chunk"]

                metadata = item["metadata"]

                st.subheader(
                    f"Source Response {idx}"
                )

                # Metadata Display
                source_name = os.path.basename(
                    metadata.get("source", "Unknown Source")
                )

                page_number = metadata.get(
                    "page",
                    "Unknown Page"
                )

                st.markdown(
                    f"""
                    **Source File:** `{source_name}`  
                    **Page:** `{page_number}`
                    """
                )

                formatted_response = re.sub(
                    r'(\d+\.)',
                    r'\n\n\1',
                    response
                )

                st.markdown(formatted_response)

                # STORE RESPONSE IN CHAT HISTORY
                assistant_message = \
                    f"Source Response {idx}:\n\n{response}"

                if assistant_message not in [
                    m["content"]
                    for m in st.session_state.messages
                    if m["role"] == "assistant"
                ]:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })

                chunk_hash = hashlib.sha256(
                    chunk.encode()
                ).hexdigest()

                feedback_key = (
                    f"{st.session_state.last_query}_"
                    f"{chunk_hash}_{idx}"
                )

                feedback_value = \
                    st.session_state.feedback_status.get(
                        feedback_key,
                        None
                    )

                col1, col2 = st.columns(2)

                with col1:

                    if feedback_value == 1:

                        st.success(
                            "👍 Positive feedback submitted"
                        )

                    else:

                        if st.button(
                                f"👍 Helpful {idx}",
                                key=f"positive_{feedback_key}"
                        ):
                            save_individual_feedback(
                                st.session_state.last_query,
                                response,
                                chunk,
                                metadata,
                                1
                            )

                            st.session_state.feedback_status[
                                feedback_key
                            ] = 1

                            st.rerun()

                with col2:

                    if feedback_value == -1:

                        st.error(
                            "👎 Negative feedback submitted"
                        )

                    else:

                        if st.button(
                                f"👎 Not Helpful {idx}",
                                key=f"negative_{feedback_key}"
                        ):
                            save_individual_feedback(
                                st.session_state.last_query,
                                response,
                                chunk,
                                metadata,
                                -1
                            )

                            st.session_state.feedback_status[
                                feedback_key
                            ] = -1

                            st.rerun()

# Run app
if __name__ == "__main__":
    main()