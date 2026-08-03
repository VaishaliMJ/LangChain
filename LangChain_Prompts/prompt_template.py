from langchain_ollama import ChatOllama
import os

###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    model = ChatOllama(
        model="llama3",
        temperature=0.0
    ) 
    response=model.invoke("Write a 5 line poem on school")
    print(response.content)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
