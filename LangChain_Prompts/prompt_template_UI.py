"""----------------------------------------------------------------------------------
    Problem Statement   :   Prompt Template With Langchain,Streamlit application 
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""


from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
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
    
    st.title("Reaserch Tool")
    paperNameInput = st.selectbox( "Select Research Paper Name", 
                                  ["Attention Is All You Need", 
                                   "BERT: Pre-training of Deep Bidirectional Transformers",
                                   "GPT-3: Language Models are Few-Shot Learners", 
                                   "Diffusion Models Beat GANs on Image Synthesis"] )

    styleInput = st.selectbox( "Select Explanation Style", 
                              ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

    lengthInput = st.selectbox( "Select Explanation Length", 
                               ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

    template = load_prompt("template.json")
    if st.button("Summarize"):
        prompt=template.invoke({
            'paperNameInput':paperNameInput,
            'styleInput':styleInput,
            'lengthInput':lengthInput
            })
        result=model.invoke(prompt)
        st.write(result.content)
    
    
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
