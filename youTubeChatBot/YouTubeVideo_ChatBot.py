"""----------------------------------------------------------------------------------
    Problem Statement   :   You Tube Video Chat Bot
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
#from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
###############################################################################
VIDEO_ID="Gfr50f6ZBvo"
BORDER="-"*60
###############################################################################
#   Function        :   getVideoTranscript
#   Input Params    :   None
#   Output Params   :   Video Transcript
#   Description     :   Returns Video Transcript
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getVideoTranscript():
    try:
        transcriptList = YouTubeTranscriptApi().fetch(VIDEO_ID)
        transcript = " ".join(chunk.text for chunk in transcriptList)
        print(transcriptList)
        """fetched_transcript = YouTubeTranscriptApi().fetch(VIDEO_ID, languages=['en'])
        #print(f"Transcripted Video chunk 0 {fetched_transcript}")
        transcript_list = fetched_transcript.to_raw_data()
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print(BORDER)
        print(transcript)
        print(BORDER)"""
        #print("Transcripted Video chunk 0 ")
    except TranscriptsDisabled:
         print("No captions available for this video.")
    except Exception as e:
        print(f"An error occurred: {e}")
            
    return transcript

###############################################################################
#   Function        :   formatTranscript
#   Input Params    :   transcriptList
#   Output Params   :   Formatted Transcript
#   Description     :   Returns Video Transcript
#   Author          :   Vaishali M Jorwekar
###############################################################################
def formatTranscript(transcriptList):
    formattedText=[
        Document(
            page_content=transcript.text,
            metadata={
                        "start": transcript.start,
                        "duration":transcript.duration
                }
        )
        for transcript in transcriptList
    ]
    
    print(f"Formatted Transcript:\n{formattedText[0]}")
    return formattedText
###############################################################################
#   Function        :   splitTranscript
#   Input Params    :   transcript
#   Output Params   :   chunks
#   Description     :   Returns chunks 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def splitTranscript(transcript):
    #cleaned_docs = [doc for doc in transcript if doc.page_content.strip()]

    spliiter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks=spliiter.create_documents([transcript])
    print(BORDER)
    print(f"Chunks:{chunks}")
    print(BORDER)
    return chunks
###############################################################################
#   Function        :   createVectorStore
#   Input Params    :   chunks,embeddingModel
#   Output Params   :   Vector Store
#   Description     :   Returns Vector Store 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def createVectorStore(chunks,embeddingModel):
    vectorStore=None
    batchSize=16
    
    for i in range(0,len(chunks),batchSize):
        batch=chunks[i : i + batchSize]
        if not vectorStore:
            vectorStore=FAISS.from_documents(batch,embeddingModel)
        else:
            vectorStore.add_documents(batch)    
    return vectorStore
###############################################################################
#   Function        :   getEmbeddings
#   Input Params    :   None
#   Output Params   :   Embeddings
#   Description     :   Returns Embeddings 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getEmbeddings():
    embeddings=OllamaEmbeddings(model="nomic-embed-text")
    #print(f"Embeddings :    {embeddings}")
    return embeddings

###############################################################################
#   Function        :   fetchLLMModel
#   Input Params    :   None
#   Output Params   :   llm Model
#   Description     :   Returns llm model object 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def fetchLLMModel():
    llm = ChatOllama(
            model="llama3",
            temperature=0
        )
    return llm
###############################################################################
#   Function        :   getPromptTemplate
#   Input Params    :   None
#   Output Params   :   prompt template
#   Description     :   Returns prompt template
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getPromptTemplate():
    prompt=PromptTemplate(
        template="""You are a helpful assistant. 
                    Answer only from the provided transcripted context.
                    If context is insufficient, return answer as You don't know !!!
                    {context}
                    Question :{question}
                    """,
        input_variables=['context','question']            
    )
    return prompt
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    transcript=getVideoTranscript()
    #formattedTranscriptText=formatTranscript(transcript)
    chunks=splitTranscript(transcript)
    
    embeddingModel=getEmbeddings()
    #vectorStore=FAISS.from_documents(chunks,embeddingModel)
    vectorStore=createVectorStore(chunks,embeddingModel)

    retriever=vectorStore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k":4}
                )
    
    llm=fetchLLMModel()    

    prompt=getPromptTemplate()

    question="what is deep mind"
    
    retrievedDocs=retriever.invoke(question)
    
    print(BORDER)
    print(BORDER)
    print(f"Retrieved Documents :\n{retrievedDocs}")
    
    print(BORDER)
    print(BORDER)
    contextText="\n\n".join(doc.page_content for doc in retrievedDocs)

    finalPrompt=prompt.invoke({
                                'context':contextText,
                               'question':question
                            })
    
    answer = llm.invoke(finalPrompt)
    
    print(f"Answer of '{question} is  : \n\n{answer.content}")
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
