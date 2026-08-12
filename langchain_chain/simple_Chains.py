"""----------------------------------------------------------------------------------
    Problem Statement   :   Langchain : Simple Chain
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""

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
    prompt = PromptTemplate(
        template="Write 5 facts about the {topic}",
        input_variables=['topic']
    )
    
    
    parser=StrOutputParser()
    
    chain=prompt | model | parser 
    
    result=chain.invoke({'topic':'Sport Cricket'})
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