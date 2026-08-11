from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from typing import List , Optional
class Movie(BaseModel):
    title:str
    release_year:Optional[int] = None
    genre : List[str]
    director : Optional[str] = None
    cast : List[str]
    rating : Optional[float] = None
    summary : str
    

parser = PydanticOutputParser(pydantic_object=Movie)







model = ChatMistralAI(
    model = "mistral-small-2506"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""
         Extract movie information from paragraph
         {format_instruction}
         
         
         """),
        
        ("human" , "{paragraph}")
    ]
)



movie_text = input("Enter Your Paragraph , description or any helpful information related to Movie =>  ")

  
final_prompt = prompt.invoke(
    {
        "paragraph": movie_text,
        "format_instruction":parser.get_format_instructions()
    }
)  


response = model.invoke(final_prompt)
movie = parser.parse(response.content)
print(movie)
