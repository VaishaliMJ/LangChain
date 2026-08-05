"""----------------------------------------------------------------------------------
    Problem Statement   :   Console Based Basic ChatBot with History Application
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
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
    chatHistory=[]
    while True:
        userInput=input("You    :   ")
        chatHistory.append(userInput)
        if userInput == "exit":
            break
        response=model.invoke(chatHistory)
        chatHistory.append(response.content)
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