from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file_to_vector_store(
    file_path: str, embeddings
) -> InMemoryVectorStore:
    """
    Loads a text file, splits it into chunks, and stores it in an in-memory vector store.

    Args:
        file_path: The path to the text file.
        embeddings: The embeddings model to use for vectorization.

    Returns:
        An InMemoryVectorStore containing the document chunks.
    """
    # 1. Load the document from the file path
    loader = TextLoader(file_path)
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    docs: List[Document] = text_splitter.split_documents(documents)

    # 3. Create and store embeddings in the in-memory vector store
    print(f"Creating vector store from {len(docs)} document chunks...")
    vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)

    return vectorstore