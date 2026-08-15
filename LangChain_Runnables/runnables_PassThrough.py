"""----------------------------------------------------------------------------------
    Problem Statement   :   Runnables : PassThrough
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
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
        template='Create a joke about {topic}',
        input_variables=['topic']
    )
    
    prompt_2=PromptTemplate(
        template='Explain the following joke : {topic}',
        input_variables=['topic']
    )
    
    jokeChain = RunnableSequence(prompt_1,model,parser)
    
    parallelChain = RunnableParallel({
        "Joke":RunnablePassthrough(),
        "explaination":RunnableSequence(prompt_2,model,parser)
        })
    
    finalChain=RunnableSequence(jokeChain,parallelChain)
    result = finalChain.invoke({'topic':'Football'})
    print(BORDER)
    print("\t\tPassThrough Sequence Output \n")
    print(BORDER)
    print(f"Joke   :   {result["Joke"]}")
    print(BORDER)
    print(f"Explaination    :   {result["explaination"]}")
    print(BORDER)

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()