"""----------------------------------------------------------------------------------
    Problem Statement   :   Runnables : Branch
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnableBranch,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
BORDER=60*"-"
###############################################################################
#   Function        :   createChatModel
#   Input Params    :   None
#   Output Params   :   chatModel
#   Description     :   Create a chat Model
#   Author          :   Vaishali M Jorwekar
###############################################################################
def createChatModel():
    model = ChatOllama(
                    model="llama3",
                    temperature=0.0
            ) 
    return model

###############################################################################
#   Function        :   wordCount
#   Input Params    :   None
#   Output Params   :   Count the number of words
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def wordCount(text):
    return len(text.split())

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
        template='Generate a detailed report on the topic {topic}',
        input_variables=['topic']
    )
    
    prompt_2 = PromptTemplate(
        template='Summarize the following topic  {text}',
        input_variables=['text']
    )
   
    
    reportGenChain = RunnableSequence(prompt_1,model,parser)
    
    branchChain = RunnableBranch(
        (lambda x : len(x.split()) > 50,RunnableSequence(prompt_2,model,parser)),
        RunnablePassthrough()
        )
    
    finalChain=RunnableSequence(reportGenChain,branchChain)
    result = finalChain.invoke({'topic':'Independence Day'})
    
    
    print(BORDER)
    print(f"\tBranch Chain Output \n{result}")
    print(BORDER)
   

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()