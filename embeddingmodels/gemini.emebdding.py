from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)

texts = (
    "Hello this is Dharmesh Sharma",
    "Hello your name is Gemini",
    "And you all are very beautiful"
)

vector = embeddings.embed_documents(texts)

print(vector)
print(len(vector))