"""----------------------------------------------------------------------------------
    Problem Statement   :   Cutom ToolKit in LangChain 
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import tool



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
    toolkit=mathToolKit()
    tools=toolkit.gettTools()
    for tool in tools:
        print(f"{tool.name} ===> {tool.description}\n")
        result = tool.invoke({"a":8, "b":3})
        print(f"Result of {tool.name} \n")
        print(f"Parameters \n  {tool.args} = {result}")
        print(BORDER)
    
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
    