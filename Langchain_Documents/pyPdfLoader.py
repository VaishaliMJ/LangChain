"""----------------------------------------------------------------------------------
    Problem Statement   :   Documents : PyPdfLoader
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

BORDER=60*"-"


###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    
   
    
    loader=PyPDFLoader("DLNotes.pdf")
    
    
    docs= loader.load()
    
    print(f"Docs Length :   {len(docs)}")
    print(BORDER)
    print(f"Page Content    :   \n{docs[0].page_content}")
    print(BORDER)
    print(f"Meta data   : \n{docs[0].metadata}")
    print(BORDER)
    


###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()