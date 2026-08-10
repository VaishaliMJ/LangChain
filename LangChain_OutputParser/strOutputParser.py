from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


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
    
    # Report prompt
    template1 = PromptTemplate(
        template="Write a detailed report on {topic}",
        input_variables=['topic']
    )
    # Summary Prompt
    template2=PromptTemplate(
        template="Write a 5 line summary on the following text./n {text} ",
        input_variables=['text']
    )
    
    prompt1=template1.invoke({"topic":"Blackhole"})
    
    result1=model.invoke(prompt1)
    
    
    prompt2 =template2.invoke({"text":result1.content})
    result2=model.invoke(prompt2)
    print(BORDER)
    print(f"Topic Information : {result1}")
    print(BORDER)
    print(f"Summary : {result2.content}")
    print(BORDER)



###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()