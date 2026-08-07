"""----------------------------------------------------------------------------------
    Problem Statement   :   Basic Langchain model invoke functionality
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""


from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated,Literal,Optional
import os


BORDER="-"*65
###############################################################################
#   Class           :   Review
#   Input Params    :   None
#   Output Params   :   chatModel
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
class Review(TypedDict):
    keyThemes     :   Annotated[list[str],"Write down all the key themes discussed in review in a list"]
    summary     :   Annotated[str,"A brief summary of the rewiew"]
    sentiment   :   Annotated[Literal["Positive","Negative"],"Return sentiment of the review"]
    pros    :   Annotated[Optional[list[str]],"Write down all pros inside a list"]
    cons    :  Annotated[Optional[list[str]],"Write down all cons inside a list"]
        
      
      
###############################################################################
#   Function        :   createChatModel
#   Input Params    :   None
#   Output Params   :   chatModel
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def createChatModel():
    model = ChatOllama(
                    model="llama3",
                    temperature=0.0
            ) 
    return model
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    model = createChatModel()
    
    structuredModel=model.with_structured_output(Review)
    
    response=structuredModel.invoke("""
                                  Machine Learning (ML) algorithms are broadly categorized into four primary types based on how they learn from data to make predictions: Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, and Reinforcement Learning.1. Supervised LearningThe model learns from labelled training data containing both inputs and the correct output answers.Regression: Predicts continuous numerical values (e.g., predicting stock prices or housing costs using Linear Regression).Classification: Assigns data into distinct, categorical buckets (e.g., sorting emails into spam/ham or classifying images using Convolutional Neural Networks (CNNs)).2. Unsupervised LearningThe model analyzes unlabelled data to discover hidden patterns, groupings, or structures on its own.Clustering: Groups similar data points together based on shared features (e.g., customer segmentation using K-Means).Dimensionality Reduction: Compresses data by removing redundant features while keeping vital info (e.g., Principal Component Analysis).Association Rule Learning: Identifies interesting relationships between variables in large databases (e.g., market basket analysis).3. Semi-Supervised LearningThe system trains on a small amount of labelled data combined with a large amount of unlabelled data.Utility: Saves massive time and cost since labelling data manually is highly expensive.Use Case: Text classification or medical imaging where only a few files are annotated by experts.4. Reinforcement LearningThe algorithm acts as an autonomous agent that learns by trial and error using a system of rewards and penalties.Mechanism: Maximizes total reward points over time by navigating an environment.Use Case: Training autonomous vehicles, robotic manufacturing arms, or mastering complex strategic games like chess and Go..
                                 """)
    
    print(BORDER)
    print(f"Summary :   {response["summary"]}")
    print(f"Sentiment   :   {response["sentiment"]}")
    print(f"Key Themes   :   {response["keyThemes"]}")
    print(f"Pros   :   {response["pros"]}")
    print(f"Cons   :   {response["cons"]}")

    print(BORDER)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
