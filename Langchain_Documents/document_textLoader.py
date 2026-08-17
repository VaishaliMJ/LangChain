"""----------------------------------------------------------------------------------
    Problem Statement   :   Documents : Text Loader
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
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
    
    loader=TextLoader("testDoc.txt",encoding="utf-8")
    
    
    docs= loader.load()
    
    
    print(f"Page Content    :   \n{docs[0].page_content}")
    print(BORDER)
    print(f"Meta data   : \n{docs[0].metadata}")
    
    
    prompt = PromptTemplate(
        template='Write a summary for following poem {poem}',
        input_variables=['poem']
    )
    
    parser =StrOutputParser()
    
    chain =  prompt | model | parser
    result= chain.invoke({'poem': docs[0].page_content})
    
    print(len(docs))
    print(BORDER)
    print(f"\tSummary of poem\n")
    print(result)
    print(BORDER)
   

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()