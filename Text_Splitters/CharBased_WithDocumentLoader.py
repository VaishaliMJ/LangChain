"""----------------------------------------------------------------------------------
    Problem Statement   :   Length Based Text Splitters,with Document Loader
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader



BORDER="-"*65



###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    loader=PyPDFLoader("TestPdfDoc.pdf")
    
    docs =loader.load()
    
    splitter=CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separator=''
    )
    
    chunks=splitter.split_documents(docs)
    for item in range(len(chunks)):
        
        print(f"Chunk No {item} :    \n{chunks[item].page_content}")
        print(BORDER)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()

