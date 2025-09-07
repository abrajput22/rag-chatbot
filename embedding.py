import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import glob
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()


embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  
    google_api_key=os.getenv('GEMINI_API_KEY'),
    transport="grpc"
)

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "pdf-knowledge-base"


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,       
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)


def ingest_pdfs(folder="pdfs"):
    pdf_files = glob.glob(f"{folder}/*.pdf")
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        docs = loader.load()
        for d in docs:
            d.metadata["pdf_file"] = os.path.basename(file)
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=60
    )
    chunks = splitter.split_documents(documents)

    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embedder.embed_query(chunk.page_content)
        vectors.append({
            "id": f"chunk-{i}",
            "values": vec,
            "metadata": {"text": chunk.page_content, "pdf_file": chunk.metadata.get("pdf_file")}
        })

    #index.upsert(vectors)
    print(f"Inserted {len(vectors)} vectors from {len(pdf_files)} PDFs")


if __name__ == "__main__":
    ingest_pdfs()

    query = "what if i am unhealthy then i have to come office or not?"
    query_vec = embedder.embed_query(query)
    results = index.query(vector=query_vec, top_k=3, include_metadata=True)
    for match in results["matches"]:
        print(match["score"], match["metadata"]["text"][:200], "...")
        print("\n")


#  python embedding.py
