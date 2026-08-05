"""----------------------------------------------------------------------------------
    Problem Statement   :   Console Based Basic Chat Bot Application
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate

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
    while True:
        userInput=input("You    :   ")
        if userInput == "exit":
            break
        response=model.invoke(userInput)
        print(f"AI   :   {response.content}")
    





###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()