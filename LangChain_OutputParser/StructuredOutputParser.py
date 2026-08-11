from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers.structured import ResponseSchema, StructuredOutputParser
BORDER="-"*60
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
                    temperature=0.7
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
    
    schema=[
        ResponseSchema(name="fact_1",description="fact_1 about the topic"),
        ResponseSchema(name="fact_2",description="fact_2 about the topic"),
        ResponseSchema(name="fact_3",description="fact_3 about the topic"),

    ]
    parser=StructuredOutputParser.from_response_schemas(schema)
    
    
    # Report prompt
    template = PromptTemplate(
        template="Give 3 facts about the {topic} \n  {format_text}",
        input_variables=["topic"],
        partial_variables={'format_text':parser.get_format_instructions()}
    )
    
    
    prompt=template.invoke({"topic":"Black hole"})
    
    result=model.invoke(prompt)
    
    parserResult=parser.parse(result.content)
    
    print(BORDER)
    print(f"LLM Result : {result}")
    print(BORDER)
    print(f"Parser Output : {parserResult}")
    print(BORDER)



###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()