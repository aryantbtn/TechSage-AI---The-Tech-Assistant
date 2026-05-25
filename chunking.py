# Fixed-Size Chunking:

from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(text)


# 2. Recursive Character Splitter:
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_text(text)


# 3. Token-Based Chunking:
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
chunks = splitter.split_text(text)


# 4. Sentence or Semantic Chunking:
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(embeddings)
chunks = splitter.split_text(text)


# 5. Document-Based Chunking:
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
).split_documents(PyPDFLoader("sample.pdf").load())