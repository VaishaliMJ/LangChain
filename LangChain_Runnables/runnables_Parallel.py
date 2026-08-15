"""----------------------------------------------------------------------------------
    Problem Statement   :   Runnables : Parallel
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
BORDER=60*"-"
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
                    temperature=0.0
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
    
    parser = StrOutputParser()
    
    prompt_1 = PromptTemplate(
        template='Generate a instagram post about {topic}',
        input_variables=['topic']
    )
    
    prompt_2=PromptTemplate(
        template='Generate a LinkedIn Post about {topic}',
        input_variables=['topic']
    )
    
    parallelChain = RunnableParallel({
        "Insta":RunnableSequence(prompt_1,model,parser),
        "linkedIn":RunnableSequence(prompt_2,model,parser)
        })
    result = parallelChain.invoke({'topic':'AI'})
    print(BORDER)
    print("\t\tParallel Sequence Output \n")
    print(BORDER)
    print(result)
    print(BORDER)
    print(BORDER)
    print(f"Instagram   :   {result["Insta"]}")
    print(BORDER)
    print(f"LinkedIn    :   {result["linkedIn"]}")
    print(BORDER)

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()