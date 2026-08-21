"""----------------------------------------------------------------------------------
    Problem Statement   :   Contextual Compression Retriever
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

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
        Document(page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world.
            Photosynthesis is the process by which green plants convert sunlight into energy.
            Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ), metadata={"source": "Doc1"}),

        Document(page_content=(
            """In medieval Europe, castles were built primarily for defense.
            The chlorophyll in plant cells captures sunlight during photosynthesis.
            Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ), metadata={"source": "Doc2"}),

        Document(page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century.
            It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
        ), metadata={"source": "Doc3"}),

        Document(page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form.
            Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
            Modern filmmaking involves complex CGI and sound design."""
        ), metadata={"source": "Doc4"})
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
   baseRetriever = vectorstore.as_retriever(search_kwargs={"k": 3})
   
   
   llm = ChatOllama(model="llama3")
   compressor = LLMChainExtractor.from_llm(llm)
   
   contextualCompressor = ContextualCompressionRetriever(
                base_retriever=baseRetriever,
                base_compressor=compressor
)
   query = "What is photosynthesis?"
   contextualResults = contextualCompressor.invoke(query)
   
   print(f"Contextual Query Results")
   for i, doc in enumerate(contextualResults):
        print(BORDER)
        print(f"\n--- Result {i+1} ---")
        print(BORDER)
        print(doc.page_content) 
        
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
