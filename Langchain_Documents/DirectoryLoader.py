"""----------------------------------------------------------------------------------
    Problem Statement   :   Documents : Directory Loader
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
#from langchain_classic.document_loaders import DirectoryLoader
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
    
    
    
    directoryLoader=DirectoryLoader(
        path="Books",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    
    
    
    docs= directoryLoader.load()
    print(f"Docs Length :   {len(docs)}")
    print(BORDER)
    
    for document in docs:
        print(f"Meta data   : \n{document.metadata}")
    
    

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()