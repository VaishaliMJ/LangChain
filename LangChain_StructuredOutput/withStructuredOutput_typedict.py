"""----------------------------------------------------------------------------------
    Problem Statement   :   Basic Langchain model invoke functionality
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""


from langchain_ollama import ChatOllama
from typing import TypedDict
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
    summary     :   str
    sentiment   :   str
      
      
      
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
                                  Artificial Intelligence (AI) and Machine Learning (ML) represent the cutting edge of modern computer science, transforming how humans interact with technology. Artificial Intelligence is the overarching science of engineering intelligent machines capable of mimicking human cognitive functions. This includes advanced reasoning, problem-solving, and natural language understanding. Machine Learning, a vital subset of AI, serves as the data-driven engine that makes this intelligence scalable. Instead of relying on rigid, pre-programmed code, ML uses mathematical algorithms to analyze vast datasets, recognize underlying patterns, and improve its own performance autonomously over time. Together, AI/ML systems power critical applications across modern society.
                                  hese include Microsoft Azure predictive healthcare diagnostics, real-time banking fraud prevention, and self-driving vehicular navigation. By automating complex tasks and unlocking insights from massive streams of data, AI and ML continue to revolutionize industries and redefine the boundaries of technological innovation.
                                 """)
    
    print(BORDER)
    print(f"Summary :   {response["summary"]}")
    print(f"Sentiment   :   {response["sentiment"]}")
    print(BORDER)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
