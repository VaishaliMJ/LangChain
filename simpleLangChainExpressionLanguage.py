import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

groqApiKey=os.getenv("GROQ_API_KEY")

model=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groqApiKey)

message=[
    SystemMessage(content="Translate the following from English to French"),#Instruction to LLM
    HumanMessage(content="Hello,How are you ?")
    
]

response=model.invoke(message)

parser=StrOutputParser()
result=parser.invoke(response)
print(result)


####Using LCEL chain the components
chain=model|parser
chainOutput=chain.invoke(message)
print(chainOutput)



#Prompt template
genericTemplate="Translate the following into {language} : "

prompt=ChatPromptTemplate.from_messages(
    [
     ("system",genericTemplate),
     ("user","{text}")
    ]
)


result=prompt.invoke({"language":"French",
               "text":"Hello"})

print(result.to_messages)



#Chain
chain=prompt|model|parser
chainOutput=chain.invoke({"language":"French","text":"Hello"})
print(chainOutput)