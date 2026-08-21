"""----------------------------------------------------------------------------------
    Problem Statement   :   Multi Query Retriever
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_ollama import ChatOllama
BORDER="-"*65

###############################################################################
#   Function        :   getDocuments
#   Input Params    :   None
#   Output Params   :   Document List
#   Description     :   Returns a list of documents
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getDocuments():
    documents = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]
    return documents
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
   documentList=getDocuments()
   embeddingModel=OllamaEmbeddings(model="nomic-embed-text") 
   vectorstore = FAISS.from_documents(
        documents=documentList,
        embedding=embeddingModel
    )
   similarityRetriever = vectorstore.as_retriever(search_type="similarity",
                                                  search_kwargs={"k": 5})
   
   
   multiqueryRetriever = MultiQueryRetriever.from_llm(
                    retriever=vectorstore.as_retriever(
                              search_kwargs={"k": 5}),
                    llm=ChatOllama(model="llama3")
)
   query = "How to improve energy levels and maintain balance?"
   similarityResults = similarityRetriever.invoke(query)
   multiqueryResults=multiqueryRetriever.invoke(query)
   
   print(f"Similarity Query Results")
   for i, doc in enumerate(similarityResults):
        print(BORDER)
        print(f"\n--- Result {i+1} ---")
        print(BORDER)
        print(doc.page_content)
   
   print("\n\n",BORDER)
   print(f"Multi Query Results")
   
   for i, doc in enumerate(multiqueryResults):
        print(BORDER)
        print(f"\n--- Result {i+1} ---")
        print(BORDER)
        print(doc.page_content)     
        
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
