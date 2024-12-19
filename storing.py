from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import asyncio
from langchain_milvus import Milvus
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

# Milvus setup
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'pdf_collection'
DIMENSION = 768  # Dimensionality of embeddings

# Connect to Milvus
connections.connect(
    alias='default',
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

# Define schema for the collection
fields = [
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
]

schema = CollectionSchema(fields, description="Embedding collection with text")

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    collection = Collection(name=COLLECTION_NAME)
else:
    collection = Collection(name=COLLECTION_NAME, schema=schema)

# Create an index for efficient searching
index_params = {
    "index_type": "IVF_FLAT",  # how the data is organized into clusters
    "metric_type": "COSINE",   #how similar the vectors are within the search
    "params": {"nlist": DIMENSION}  #number of clusters.
}
collection.create_index(field_name="embedding", index_params=index_params)

embed_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Process the PDF 
async def process_pdf(file_path):
    loader = PyPDFLoader(
        file_path=file_path,
        extract_images=True,
    )

    docs = []
    
    # Iterate through the documents from the PDF
    async for doc in loader.alazy_load():
        docs.append(doc)
    
    all_splits = chunk_documents(docs)

    insert_documents_to_milvus(all_splits)


# Chunking
def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200, 
        add_start_index=True,   # index where it begins in the original text.
    )

    all_splits = text_splitter.split_documents(docs)
    return all_splits

# Embed chunks and insert into Milvus
def insert_documents_to_milvus(docs):
    chunks = [doc.page_content for doc in docs]
    
    embeddings = embed_model.encode(chunks)

    # Insert the embeddings and text into Milvus
    collection.insert([embeddings.tolist(), chunks])

    collection.flush()
    collection.load()

file_path = "./docs/financial-terms.pdf"
asyncio.run(process_pdf(file_path))

