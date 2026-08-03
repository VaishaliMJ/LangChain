from langchain_ollama import ChatOllama
import streamlit as st

###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    st.title("Reaserch Tool")
    userInput=st.text_input("Enter your prompt")
    
    model = ChatOllama(
        model="llama3",
        temperature=0.0
    ) 
    if st.button("Summarize"):
        response=model.invoke(userInput)
        st.write(response.content)
    
    
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
