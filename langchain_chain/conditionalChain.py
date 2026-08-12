"""----------------------------------------------------------------------------------
    Problem Statement   :   Langchain : Conditional Chain
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda

BORDER="-"*60
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
                temperature=0
            ) 
   
    return model
###############################################################################
#   class           :   Feedback
#   Input Params    :   None
#   Output Params   :   chatModel
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
class Feedback(BaseModel):
    
    sentiment : Literal["Positive","Negative"]=Field(description="Give Sentiment of the feedback")
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    model = createChatModel()
    parser=StrOutputParser()
    
    feedbackParser=PydanticOutputParser(pydantic_object=Feedback)
    
    prompt=PromptTemplate(
        template=("You are an product reviewer.\n"
        "Task: Analyze the user product review enclosed in triple quotes below and evaluate its sentiment.\n"
        "Constraint: Do not generate replies, templates, or chat responses. Output raw JSON only.\n\n"
        "User review to evaluate:\n"
        '"""\n{feedback}\n"""\n\n'
        "{format_instructions}"
        ),
        input_variables=["feedback"],
        partial_variables={'format_instructions': feedbackParser.get_format_instructions()}
    )
    
    classifierChain= prompt | model | feedbackParser
    
    #result = classifierChain.invoke({'feedback':"This phone is excellent"})
    #print(BORDER)
    #print(f"Feedback Sentiment  : {result}")
    #print(f"Actual Sentiment    :   {result.sentiment}")
    
    positivePrompt=PromptTemplate(
        template="Write an appropriate response to this  positive feedback \n{feedback}",
        input_variables=['feedback']
    )
    
    negativePrompt=PromptTemplate(
        template="Write an appropriate response to this negative feedback \n{feedback}",
        input_variables=['feedback']
    )
    
    branchChain=RunnableBranch(
        (lambda x : x.sentiment == 'Positive',positivePrompt | model | parser),
        (lambda x : x.sentiment == 'Negative',negativePrompt | model | parser),
        RunnableLambda(lambda x : "Could not find sentiment")
    )
    
    
    chain = classifierChain | branchChain
    result=chain.invoke({'feedback':"This smartphone is excellent"})
    print(BORDER)
    print(f"Feedback Sentiment  : {result}")
    chain.get_graph().print_ascii()
    
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()