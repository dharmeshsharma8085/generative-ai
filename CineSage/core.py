from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model = "mistral-small-2506"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an information extraction AI.

Analyze the following movie paragraph and extract useful information from it.

Extract:

- Movie Name
- Genre
- Main Characters
- Cast (only if mentioned)
- Director (only if mentioned)
- Release Year (only if mentioned)
- Setting
- Plot
- Main Themes
- Central Conflict
- Key Relationships
- Overall Tone
- Quick Summary

Rules:

- Extract only information supported by the paragraph.
- Do not guess or invent missing information.
- If something is not mentioned, write "Not mentioned".
- Keep the extracted information concise.
- Give the Quick Summary in 2-3 sentences.

Return the information in a clean, readable format.
"""
    ),
    (
        "human",
        """
        Extract information from paragrapgh:
        
        {paragraph}
        """
    )
])


movie_text = input("Enter Your Paragraph , description or any helpful information related to Movie =>  ")

  
final_prompt = prompt.invoke(
    {
        "paragraph": movie_text
    }
)  


response = model.invoke(final_prompt)
print(response.content)
