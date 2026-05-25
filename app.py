import json
import streamlit as st
from transformers import pipeline
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Load model once (important for performance)
@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=-1
    )
    # print(pipe("Explain RAG in simple terms", max_new_tokens=50))
    return pipe

pipe = load_model()


# Chunking flow
# Step1 - Load
loader = PyPDFLoader("pdf/IT-Support.pdf")
documents = loader.load()


# Step 2 - Chunking Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# Ste3 - Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step4 - VectorDB Store
vectorstore = FAISS.from_documents(chunks, embeddings)


def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_rag_response(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])


    prompt = f"""
        You are an expert IT support assistant.
        
        Follow these rules strictly:
        1. Answer ONLY using the provided context.
        2. If the answer is not found, say: "I don't have enough information."
        3. Provide step-by-step solutions.
        4. Be clear and concise.
        
        Context:
        {context}
        
        User Question:
        {query}
        
        Final Answer (step-by-step):
        """

    result = pipe(
        prompt,
        max_new_tokens=150,
        temperature=0.5
    )

    return result[0]['generated_text']

def load_json_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"
        docs.append(Document(page_content=content))

    return docs

def split_documents(docs):
    return docs  # No splitting needed

@st.cache_resource
def setup_rag():
    docs = load_json_documents("pdf/json-question-for-model-training.json")  # your file
    db = create_vectorstore(docs)
    return db

vectorstore = setup_rag()

def get_llama_response(input_text, max_words):
    prompt = f"""
You are an intelligent assistant.

Analyze the user query and decide the best format for the response.

Guidelines:
- If the query asks "how", give step-by-step instructions.
- If it is a conceptual question, give a clear explanation.
- If it is broad or topic-based, write a structured blog.
- If it is simple, give a concise answer.
- Use bullet points where helpful.
- Keep response within ~{max_words} words.

User Query:
{input_text}

Final Answer:
"""

    result = pipe(
        prompt,
        max_new_tokens=int(max_words),
        do_sample=True,
        temperature=0.6
    )

    output = result[0]['generated_text']
    return output.replace(prompt, "").strip()

# Streamlit UI
st.title("AI Samadhan (Chatbot)")
mode = st.selectbox("Choose Mode", ["Ask Questions", "Generate Blog"])

if mode == "Ask Questions":
    input_text = st.text_input("Ask your question")
else:
    input_text = st.text_input("Enter blog topic")

no_words=None

if mode == "Generate Blog":
    no_words = st.slider("Number of Words", 50, 300, 100)

if st.button("Submit"):

    if input_text.strip() == "":
        st.warning("Please enter input")

    else:
        if mode == "Ask Questions":
            output = get_rag_response(input_text, vectorstore)

        else:  # Generate Blog
            output = get_llama_response(input_text, no_words)

        st.write(output)

# if st.button("Generate"):
#     if input_text.strip() == "":
#         st.warning("Please enter a topic")
#     else:
#         output = get_llama_response(input_text, no_words)
#         st.write(output)







'''
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def getLLAmarRsponse(input_text, no_words, blog_style):
    llm = CTransformers(model='model-00001-of-00055.safetensors', model_type="llama", config = {'max_new_tokens': 256, 'temperature': 0.01})

    template=f"""
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words
    """

    prompt = PromptTemplate(input_variables=["style", "text", "n_words"], template=template)

    response=llm(prompt.format(style = blog_style, text= input_text, n_words= no_words))
    print(response)
    return response
    pass


st.set_page_config(page_title="Generate Blogs", page_icon="🤖", layout="centered", initial_sidebar_state="collapsed")
st.header("Generate Blogs 🤖")
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No of words")

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', "Data Scientist", 'Commonn People'), index=0)

submit = st.button("Generate")

# Final test
if submit:
    st.write(getLLAmarRsponse(input_text, no_words, blog_style))
'''