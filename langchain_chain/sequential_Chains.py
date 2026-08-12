from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    
    # Report prompt
    prompt_1 = PromptTemplate(
        template="Generate a detailed information on  {topic}",
        input_variables=['topic']
    )
    
    prompt_2=PromptTemplate(
        template="Generate a 5 poniter summary from the following text \n {text}",
        input_variables=["text"]
    )
    
    parser=StrOutputParser()
    
    chain=prompt_1 | model | parser | prompt_2 | model | parser 
    
    result=chain.invoke({'topic':'Football'})
    print(BORDER)
    print(f"Summary : {result}")
    print(BORDER)
    print("Chain Visualisation")
    print(BORDER)
    chain.get_graph().print_ascii()


###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()