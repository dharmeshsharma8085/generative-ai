from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    temperature=1.0,
    max_tokens=20
)

response = model.invoke(
    "Give me a paragraph about what is machine learning?"
)

print(response.content)