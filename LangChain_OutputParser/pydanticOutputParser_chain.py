from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

BORDER="-"*60
###############################################################################
#   Function        :   createChatModel
#   Input Params    :   None
#   Output Params   :   chatModel
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def createChatModel():
    model = ChatOllama(
                    model="llama3",
                    temperature=0.7
            ) 
    return model

###############################################################################
#   Class           :   Person
#   Input Params    :    None
#   Output Params   :   pydentic object
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt=18,description="Age of the person")
    city : str = Field(description="Name of the city to which the person belongs")
    
    
      
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    
    
    model=createChatModel()
    
    parser=PydanticOutputParser(pydantic_object=Person)
    
    # Report prompt
    template = PromptTemplate(
        template="Generate name,age and city of a fictional {place} person \n {format_text}",
        input_variables=["place"],
        partial_variables={'format_text':parser.get_format_instructions()}
    )
    
    
    chain= template | model | parser
    result=chain.invoke({"place":"USA"})
    print(BORDER)
    print(f"LLM Result : {result}")
    print(BORDER)
    



###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()