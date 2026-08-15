"""----------------------------------------------------------------------------------
    Problem Statement   :   Runnables : Lambda
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda
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
        template='Write a joke about {topic}',
        input_variables=['topic']
    )
    
   
    
    jokeChain = RunnableSequence(prompt_1,model,parser)
    
    parallelChain = RunnableParallel({
        "Joke":RunnablePassthrough(),
        "WordCount":RunnableLambda(wordCount)
        })
    
    finalChain=RunnableSequence(jokeChain,parallelChain)
    result = finalChain.invoke({'topic':'AI'})
    
    finalResult=f"{result['Joke']} \n word Count -{result['WordCount']}"
    print(BORDER)
    print(f"\tRunnable Lambda Output \n{finalResult}")
    print(BORDER)
   

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()