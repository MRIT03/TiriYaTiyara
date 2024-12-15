import os
from dotenv import load_dotenv

from langchain_chroma import Chroma  # Vector database for storing embeddings
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

# Set your Google API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# Initialize the embedding model using Google's generative AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Initialize (or load) your Chroma vector store
vector_store = Chroma(
    collection_name="embeddings",
    embedding_function=embeddings,
    persist_directory="./vector_db",
    collection_metadata={"hnsw:space": "cosine"}
)

# Path to your CSV file
csv_file_path = "updated_dataset_with_avg_price.csv"

# Load data from CSV into documents (with UTF-8 encoding if needed)
loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
documents = loader.load()

print("I am alive")
# Add documents in smaller batches to avoid exceeding batch size limits
batch_size = 50
i = 0
print(len(documents))
while True:
    print("Batch " + str(i))
    print(len(documents))
    batch = documents[i:i+batch_size]
    vector_store.add_documents(batch)
    time.sleep(5)
    i = i + batch_size


# (Optional) Example similarity search:
# results = vector_store.similarity_search(query="Explain the concept of matrix multiplication", k=3)
# for i, doc in enumerate(results, start=1):
#     print(f"Result {i}:\n{doc.page_content}\n")
