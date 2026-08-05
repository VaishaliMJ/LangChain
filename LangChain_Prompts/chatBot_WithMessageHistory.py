"""----------------------------------------------------------------------------------
    Problem Statement   :   Console Based Basic ChatBot with History Application
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

BORDER=60*"-"
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
    
    model=createChatModel()
    chatHistory=[
        SystemMessage(content="You are a helpful assistant"),
    ]
    while True:
        userInput=input("You    :   ")
        chatHistory.append(HumanMessage(content=userInput))
        if userInput == "exit":
            break
        response=model.invoke(chatHistory)
        chatHistory.append(AIMessage(content=response.content))
        print(f"AI   :   {response.content}")
    
    
    print(BORDER)
    print("\t\tConversation Summary ")
    print(BORDER)
    print(chatHistory)
    print(BORDER)





###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()