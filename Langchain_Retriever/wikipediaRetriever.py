"""----------------------------------------------------------------------------------
    Problem Statement   :   Wikipedia Retriever
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.retrievers import WikipediaRetriever
import wikipedia
# Add a unique user agent to satisfy Wikimedia's API policy
wikipedia.set_user_agent("MyLangChainApp/1.0 (contact: email@example.com)")


BORDER="-"*65

###############################################################################
#   Function        :   getRetriever
#   Input Params    :   None
#   Output Params   :   retriever object
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getRetriever(retrieverName):
    retriever = retrieverName(top_k_results=2, lang="en")
    return retriever
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    #retriever = getRetriever(WikipediaRetriever)
    retriever = WikipediaRetriever(top_k_results=2, lang="en")
    # Define your query
    query = "History of cricket"

    # Get relevant Wikipedia documents
    result = retriever.invoke(query)
    print(BORDER)
    print(f"Result From Wikipedia   :   \n{result}")
    
    for i,doc in enumerate(result):
        print(BORDER)
        print(f"Result  {i+1}")
        print(BORDER)
        print(f"Content :   \n{doc.page_content}")

###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
