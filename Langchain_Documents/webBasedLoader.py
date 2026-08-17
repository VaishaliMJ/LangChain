"""----------------------------------------------------------------------------------
    Problem Statement   :   Documents : Directory Loader
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
#from langchain_classic.document_loaders import DirectoryLoader
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
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    
    url="https://www.kaggle.com/code/chanchal24/titanic-dataset"    
    loader=WebBaseLoader(url)
    model=createChatModel()
    
    
    parser=StrOutputParser()
    
    prompt = PromptTemplate(
             template='Answer the question  \n {question} from following {text}',
            input_variables=['question','text']
    )
    
    docs= loader.load()
    
    
    chain =  prompt |  model | parser
    result = chain.invoke({'question':"What is dataset name ",
                           "text":docs[0].page_content})
    
    
    print(BORDER)
    print(f"Length docs :   {len(docs)}")
    print(BORDER)
    print(f"Meta data   : \n{docs[0].metadata}")

    print(BORDER)
    print(f"Page Content : \n{docs[0].page_content}")
    
    print(f"Result:{result}")

    
    
    

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()