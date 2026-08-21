"""----------------------------------------------------------------------------------
    Problem Statement   :   Maximum Marginal Relevance
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

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
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
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
   similarityRetriever = vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": 3,"lambda_mult":0.5})
   
   
   
   query = "What is langchain?"
   similarityResults = similarityRetriever.invoke(query)
   
   print(f"Similarity Query Results")
   for i, doc in enumerate(similarityResults):
        print(BORDER)
        print(f"\n--- Result {i+1} ---")
        print(BORDER)
        print(doc.page_content) 
        
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
