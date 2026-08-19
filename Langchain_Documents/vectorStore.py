"""----------------------------------------------------------------------------------
    Problem Statement   :   Semantic Based
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BORDER="-"*65
###############################################################################
#   Function        :   docs
#   Input Params    :   None
#   Output Params   :   documents
#   Description     :   Document List
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getDocuments():
    doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
    doc2 = Document(
            page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
            metadata={"team": "Mumbai Indians"}
        )
    doc3 = Document(
            page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
            metadata={"team": "Chennai Super Kings"}
        )
    doc4 = Document(
            page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
            metadata={"team": "Mumbai Indians"}
        )
    doc5 = Document(
            page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
            metadata={"team": "Chennai Super Kings"}
        )
    return [doc1,doc2,doc3,doc4,doc5]
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    docs=getDocuments()
    vectorStore=Chroma(
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="myChromaDB",
        collection_name="sample"
    )
    
    vectorStore.add_documents(docs)
    print(f"Documents: {vectorStore.get(include=['embeddings','documents', 'metadatas'])}")
    
    # search documents
    result= vectorStore.similarity_search(
                query='Who among these are a bowler?',
                k=2
                )
    print(BORDER)
    print(BORDER)
    print(f"Search Result   :   {result}")
    
    
    # search with similarity score
    result= vectorStore.similarity_search_with_score(
            query='Who among these are a bowler?',
            k=2
            )
    
    print(BORDER)
    print(BORDER)
    print(f"Search Result   :   {result}")
    
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
