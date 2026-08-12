from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

BORDER="-"*60
###############################################################################
#   Function        :   createChatModel
#   Input Params    :   apiKey
#   Output Params   :   chatModel
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def createChatModel(apiKey):
    if apiKey=="":
        model = ChatOllama(
                        model="llama3",
                        temperature=0.7
            ) 
    else:
            GROQ_API_KEY=apiKey
            model=ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=GROQ_API_KEY)  
    return model
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    
    load_dotenv()
    model_1=createChatModel(apiKey="")
    model_2=createChatModel(apiKey=os.getenv("GROQ_API_KEY"))
    
    # Report prompt
    prompt_1 = PromptTemplate(
        template="Generate short and simple notes from following {text}",
        input_variables=['text']
    )
    
    prompt_2=PromptTemplate(
        template="Generate a 5 short questions and snaswers from the following text \n {text}",
        input_variables=["text"]
    )
    
    prompt_3 =PromptTemplate(
        template="Merge the provided notes and quiz into a single document \n Notes : {notes} \n Quiz : {quiz}",
        input_variables=["notes","quiz"]
    )
    parser=StrOutputParser()
    
    paralllel_chain=RunnableParallel({
        "notes": prompt_1 | model_1 | parser,
        "quiz":prompt_2 | model_2 | parser
        }
        
    )
    
    merge_chain= prompt_3 | model_1 | parser
    
    chain=paralllel_chain | merge_chain
    
    text="""K-Means Clustering groups similar data points into clusters without needing labeled data. It is used to uncover hidden patterns when the goal is to organize data based on similarity.

    Helps identify natural groupings in unlabeled datasets
    Works by grouping points based on distance to cluster centers
    Commonly used in customer segmentation, image compression and pattern discovery
    Useful when you need structure from raw, unorganized data
    Working of K-Means Clustering
    Suppose we are given a data set of items with certain features and values for these features like a vector. The task is to categorize those items into groups. To achieve this we will use the K-means algorithm. "
    k
    k" represents the number of groups or clusters we want to classify our items into.

    The algorithm will categorize the items into "
    k" groups or clusters of similarity. To calculate that similarity we will use the Euclidean distance as a measurement. The algorithm works as follows:  
    Initialization: We begin by randomly selecting k cluster centroids.
    Assignment Step: Each data point is assigned to the nearest centroid, forming clusters.
    Update Step: After the assignment, we recalculate the centroid of each cluster by averaging the points within it.
    Repeat: This process repeats until the centroids no longer change or the maximum number of iterations is reached.
    The goal is to partition the dataset into 
    k
    k clusters such that data points within each cluster are more similar to each other than to those in other clusters.    
        """
        
    result=chain.invoke({'text':text})
    print(BORDER)
    print(f"Summary : {result}")
    
    chain.get_graph().print_ascii()


###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()