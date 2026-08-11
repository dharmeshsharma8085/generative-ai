from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

texts =(
    "Hello this is Dharmesh Sharma",
    "Hello your name is MistralAI",
    "And you all are very beautiful"
)

vector = embeddings.embed_documents(texts)

print(vector)
print(len(vector))