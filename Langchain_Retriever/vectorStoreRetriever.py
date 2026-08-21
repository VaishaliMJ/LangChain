"""----------------------------------------------------------------------------------
    Problem Statement   :   Vector StoreRetriever
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

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
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
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
   vectorstore = Chroma.from_documents(
        documents=documentList,
        embedding=embeddingModel,
        collection_name="my_collection"
    )
   retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
   query = "What is Chroma used for?"
   results = retriever.invoke(query)
   
   for i, doc in enumerate(results):
        print(BORDER)
        print(f"\n--- Result {i+1} ---")
        print(BORDER)
        print(doc.page_content)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
