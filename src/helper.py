from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob = "**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_text(docs: List[Document]) -> List[Document]:
    """ 
    Given list of Document objects, return a new list of document objects
    containing only 'source' in metadata and the original page_content.

    """
    minimal_docs:List[Document]=[]
    for doc in docs:
        src =doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function =len
    )
    texts = text_splitter.split_documents(minimal_docs)
    return texts

def download_embeddinngs():
    """
    Download and return the HuggingFace embeddding model."""
    model_name ="sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name
    )
    return embedding
