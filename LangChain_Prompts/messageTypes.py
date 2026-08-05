"""----------------------------------------------------------------------------------
    Problem Statement   :   Message Types
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
    messages=[
        SystemMessage(content="You are a helpful assistant").content,
        HumanMessage(content="Tell me about LLM's").content
    ]
    result=model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print(BORDER)
    print("\t\tConversation Summary ")
    print(BORDER)
    print(messages)
    print(BORDER)





###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()