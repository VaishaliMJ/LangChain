"""----------------------------------------------------------------------------------
    Problem Statement   :   Semantic Based
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings



BORDER="-"*65



###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    text="""
    Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


    Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.


    """

    
    textSplitter=SemanticChunker(
        OllamaEmbeddings(model="nomic-embed-text"),
        breakpoint_threshold_type="standard_deviation",
        breakpoint_threshold_amount=1
        
    )
    
    chunks=textSplitter.create_documents([text])
    print(f"Number Of Chunks  : {len(chunks)}")
    print(BORDER)
    print(BORDER)
    print(chunks)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()

