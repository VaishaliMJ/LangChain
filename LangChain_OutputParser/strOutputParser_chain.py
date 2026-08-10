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
    template1 = PromptTemplate(
        template="Write a detailed report on {topic}",
        input_variables=['topic']
    )
    # Summary Prompt
    template2=PromptTemplate(
        template="Write a 5 line summary on the following text./n {text} ",
        input_variables=['text']
    )
    
    parser=StrOutputParser()
    
    chain=template1 | model | parser | template2 | model | parser
    
    result=chain.invoke({'topic':'blackhole'})
    print(BORDER)
    print(f"Summary : {result}")
    print(BORDER)



###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()