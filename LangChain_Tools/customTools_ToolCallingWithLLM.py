"""----------------------------------------------------------------------------------
    Problem Statement   :   Toll calling with LLM
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage
from  langchain_ollama import ChatOllama



BORDER = "-"*65

###############################################################################
#   Function        :   multiply
#   Input Params    :   a,b
#   Output Params   :   None
#   Description     :   Multiply Function
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def multiply(a : int,b : int)->int:
    """Multiplies two numbers"""
    return a * b    
###############################################################################
#   Function        :   add
#   Input Params    :   a,b
#   Output Params   :   None
#   Description     :   Addition Function
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def add(a : int,b : int)->int:
    """Adds two numbers"""
    return a + b    
###############################################################################
#   Function        :   subtract
#   Input Params    :   a,b
#   Output Params   :   None
#   Description     :   Subtraction Function
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def subtract(a : int,b : int)->int:
    """Subtract two numbers"""
    return a - b    

###############################################################################
#   Class           :   mathToolKit
#   Input Params    :   a,b
#   Output Params   :   None
#   Description     :   Addition Function
#   Author          :   Vaishali M Jorwekar
###############################################################################
class mathToolKit:
    def gettTools(self):
       return[add,subtract,multiply] 
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    llmModel=ChatOllama(model="llama3.1", temperature=0)
    result=llmModel.invoke("hi")
    print(f"Result  :{result.content}")
    llmWithTools=llmModel.bind_tools([multiply])
    query = HumanMessage("Can you perfrom arithmetic operations on 67 and 36")
    
    messages=[query]
    result=llmWithTools.invoke(messages)
    messages.append(result)
    
    print(BORDER)
    print(f"Messages : \n{messages}")
    toolResult=multiply.invoke(result.tool_calls[0])
    print(f"Tool result :   \n{toolResult}")
    
    messages.append(toolResult)
    print(BORDER)
    print(f"Messages : \n{messages}")
    print(BORDER)
    
    result=llmWithTools.invoke(messages).content
    print(BORDER)
    print(f"Results : \n{result}")
    print(BORDER)
    
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
    