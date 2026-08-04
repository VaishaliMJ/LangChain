from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate
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

    template = PromptTemplate(
        template="""
            Please summarize the research paper titled "{paperNameInput}" with the following specifications:
            Explanation Style: {styleInput}  
            Explanation Length: {lengthInput}  
            1. Mathematical Details:  
            - Include relevant mathematical equations if present in the paper.  
            - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
            2. Analogies:  
            - Use relatable analogies to simplify complex ideas.  
            If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
            Ensure the summary is clear, accurate, and aligned with the provided style and length.
            """,
            input_variables=['paperNameInput', 'styleInput','lengthInput'],
            validate_template=True
            )
    
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
