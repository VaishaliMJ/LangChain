"""
    Problem Statement   :   Summarize a document using LangChain's Map And Reduce document summarization chain.
    
    Name                :   Vaishali M. Jorwekar
    
    Date                :   25 Nov 2026
    
    Description         :   This script demonstrates how to summarize a document using LangChain's 
                            
                            Map And Reduce method

    """
##################################################################################################
from langchain_community.document_loaders import PyPDFLoader
#from pypdf import PdfReader
#from langchain import PromptTemplate
#from langchain.chains.summarize import load_summarize_chain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

##################################################################################################
#   Function Name   :   loadDocuments
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Loads the PDF document
#   Name            :   Vaishali M. Jorwekar
#   Date            :   25 Nov 2026
##################################################################################################
def loadDocuments():
    loader=PyPDFLoader("apjspeech.pdf")
    pages=loader.load_and_split()
    #print(pages)
    return pages
##################################################################################################
#   Function Name   :   createPromptTemplate
#   Input Params    :   text
#   Output Params   :   None
#   Description     :   Create Prompt template
#   Name            :   Vaishali M. Jorwekar
#   Date            :   25 Nov 2026
##################################################################################################
def createPromptTemplate(text):
    template="""
                Write a concise and short summary of the following speech
                Speech:{text}
                """
    prompt=PromptTemplate(input_variables=['text'],
                          template=template)
    return prompt
##################################################################################################
#   Function Name   :   summaryPromptTemplate
#   Input Params    :   text
#   Output Params   :   None
#   Description     :   Create Summary Prompt template
#   Name            :   Vaishali M. Jorwekar
#   Date            :   25 Nov 2026
##################################################################################################
def summaryPromptTemplate(text):
    template="""
                Write a final summary of the entire speech with these important points.
                Add a motivational title,Start the precise summary with an introduction
                and provide the summary in number points for the speech.
                Speech:{text}
                """
    prompt=PromptTemplate(input_variables=['text'],
                          template=template)
    return prompt

##################################################################################################
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   main function
#   Name            :   Vaishali M. Jorwekar
#   Date            :   25 Nov 2026
##################################################################################################
def main():
    
    docPages=loadDocuments()
    
    finalDocs=RecursiveCharacterTextSplitter(
                            chunk_size=2000,
                            chunk_overlap=100
                        ).split_documents(docPages)
    
    prompt=createPromptTemplate(docPages)
    
    llm=ChatOllama(model="llama3")
    
    finalPrompt=summaryPromptTemplate(docPages)
    
    summaryChain=load_summarize_chain(
                            llm=llm,
                            chain_type="map_reduce",
                            map_prompt=prompt,
                            verbose=True,
                            combine_prompt=finalPrompt
                            )
    
    
    docSummary=summaryChain.run(finalDocs)
    
    print("################################################################")
    print("\t\t\tDocument Summary")
    print("################################################################")
    print(docSummary)
##################################################################################################
# Starting Point
##################################################################################################
if __name__=="__main__":
    main()    