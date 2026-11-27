"""
--------------------------------------------------------------------------------------------
    Problem Statement   :   Summarize a You Tube video or Web Site Content
    
    Name                :   Vaishali M. Jorwekar
    
    Date                :   27 Nov 2026
    
    Description         :   This Streamlit app demonstrates the You Tube Video and 
                            
                            Web Site Content Summarization
--------------------------------------------------------------------------------------------
"""
#   Imports
import streamlit as st
from langchain_ollama import ChatOllama
import validators
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
#from langchain_community.document_loaders import WebPageLoader
from langchain_community.document_loaders import UnstructuredURLLoader

from langchain_community.document_loaders.youtube import YoutubeLoader

import traceback

##################################################################################################
#   Function Name   :   createPromptTemplate
#   Input Params    :   text
#   Output Params   :   None
#   Description     :   Create Prompt template
#   Name            :   Vaishali M. Jorwekar
#   Date            :   27 Nov 2026
##################################################################################################
def createPromptTemplate(text):
    template="""
                Provide a summary of the following content in 400 words
                Content : {text}
                """
    prompt=PromptTemplate(input_variables=['text'],
                          template=template)
    return prompt
##################################################################################################
#   Function Name   :   clearUrl
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Clear URL 
#   Name            :   Vaishali M. Jorwekar
#   Date            :   27 Nov 2026
##################################################################################################
def clearUrl():
    st.session_state.url = ""
##################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   main function
#   Name            :   Vaishali M. Jorwekar
#   Date            :   27 Nov 2026
##################################################################################################
def main():
    if "url" not in st.session_state:
        st.session_state.url = ""
    st.title("🦜 You Tube Video And Web Site Summarization Using Langchain ")
    st.subheader("Summarize Video or Web Content")
    #   LLM Model
    llm=ChatOllama(model="llama3")
    
    
    col1,col2=st.columns([3,1]) 
    with col1:  
        url=st.text_input("URL",label_visibility="collapsed",key="url")
    with col2:
            st.button(
            "Clear URL",
            on_click=clearUrl
            )
    if st.button("Summarize the content from You Tube or Web Site"):
        #   Validate URL
        if not url.strip():
            st.error("Please enter the URL")
        elif not validators.url(url):
            st.error("Please enter valid URL")  
        else:
            try:
                with st.spinner("Waiting...."):
                    if "youtube.com" in url:
                        loader=YoutubeLoader.from_youtube_url(url,add_video_info=False)
                    else:
                        loader=UnstructuredURLLoader(urls=[url],ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})    
                    data=loader.load()
                    #  Prompt
                    prompt=createPromptTemplate(data)  
                    
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    
                    summary=chain.run(data)
                    
                    st.success(summary)
            except Exception as e:
                 st.error(f"Error: {str(e)}")
                 st.code(traceback.format_exc())
              
##################################################################################################
#   Entry Point of the program
##################################################################################################
if __name__=="__main__":
    main()    