from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    
    
    model=createChatModel()
    parser=JsonOutputParser()
    
    # Report prompt
    template = PromptTemplate(
        template="Give me name,age,city of a cricketer \n  {format_text}",
        input_variables=[],
        partial_variables={'format_text':parser.get_format_instructions()}
    )
    
    
    chain=template | model | parser
    result= chain.invoke({})
    print(BORDER)
    print(f"LLM Result : {result}")
    print(BORDER)
    



###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()