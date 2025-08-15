from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_text,text_split,download_embeddinngs
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
pineconeapi_key = PINECONE_API_KEY

pc = Pinecone(api_key = pineconeapi_key)

extracted_data = load_pdf_files(data ="Data/")
filter_data = filter_text(extracted_data)
text_chunk = text_split(filter_data)
embedding = download_embeddinngs()


index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
   embedding=embedding,
    index_name = index_name
)

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name = index_name
) 