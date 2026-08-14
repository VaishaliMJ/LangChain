"""----------------------------------------------------------------------------------
    Problem Statement   :   Runnables : Sequence
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence
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
        template='Write few lines  about{topic}',
        input_variables=['topic']
    )
    
    prompt_2=PromptTemplate(
        template='Explain the following {topic}',
        input_variables=['topic']
    )
    
    chain = RunnableSequence(prompt_1,model,parser,prompt_2,model,parser)
    result = chain.invoke({'topic':'What is Machine Learning?'})
    print(BORDER)
    print("\t\tLLM Output ")
    print(BORDER)
    print(result)
    print(BORDER)

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()