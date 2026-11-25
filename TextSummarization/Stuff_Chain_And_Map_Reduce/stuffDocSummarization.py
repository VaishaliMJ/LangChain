"""
    Problem Statement   :   Summarize a document using LangChain's stuff document summarization chain.
    
    Name                :   Vaishali M. Jorwekar
    
    Date                :   25 Nov 2026
    
    Description         :   This script demonstrates how to summarize a document using LangChain's stuff docu

    """
##################################################################################################
from langchain_community.document_loaders import PyPDFLoader
#from pypdf import PdfReader
#from langchain import PromptTemplate
#from langchain.chains.summarize import load_summarize_chain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
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
#   Function Name   :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   main function
#   Name            :   Vaishali M. Jorwekar
#   Date            :   25 Nov 2026
##################################################################################################
def main():
    print("Hello")
    docPages=loadDocuments()
    prompt=createPromptTemplate(docPages)
    llm=ChatOllama(model="llama3")
    chain=load_summarize_chain(
                            llm=llm,
                            chain_type="stuff",
                            prompt=prompt,
                            verbose=True
                            )
    docSummary=chain.run(docPages)
    print("################################################################")
    print("\t\t\tDocument Summary")
    print("################################################################")
    print(docSummary)
##################################################################################################
# Starting Point
##################################################################################################
if __name__=="__main__":
    main()    