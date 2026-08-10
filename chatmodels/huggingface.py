from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

load_dotenv()

model = HuggingFaceEndpoint(
    #repo_id="YOUR_MODEL_ID",
    temperature=1.0,
    max_new_tokens=20
)

response = model.invoke(
    [HumanMessage(content="Give me a paragraph about what is machine learning?")]
)

print(response.content)