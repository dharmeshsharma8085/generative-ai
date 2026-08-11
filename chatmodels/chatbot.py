from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage , HumanMessage, AIMessage

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

print("---------------- Welcome ----------------")
print("Type 0 to exit the application")
print("------------------------------------------")

messages = [
    SystemMessage(content = "you are a  funny AI agent")]

while True:

    prompt = input("You: ")

    if prompt == "0":
        print("Goodbye! 👋")
        break

    
    messages.append(HumanMessage(content=prompt))

   
    response = model.invoke(messages)

 
    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)
    

print(messages)