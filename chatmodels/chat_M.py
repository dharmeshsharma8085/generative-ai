# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# load_dotenv()

# model = init_chat_model(
#     "gemini-3.5-flash",
#     model_provider="google_genai",
#     temperature = 1.0
# ) # chat model

                 

# # model class
# #from langchain_google_genai import ChatGoogleGenerativeAI
# #model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")  model class
# response = model.invoke("Give me a pargraph of What is machine learning?")
# print(response.content)
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "mistral-large-latest",
    model_provider="mistralai",
    temperature=1.0
)  # chat model


# model class

# from langchain_mistralai import ChatMistralAI
# model = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=1.0
# )  # model class

response = model.invoke("Give me a paragraph about what is machine learning?")
print(response.content)