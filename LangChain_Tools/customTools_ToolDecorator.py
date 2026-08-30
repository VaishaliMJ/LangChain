"""----------------------------------------------------------------------------------
    Problem Statement   :   Cutom Tools LangChain :  Using Tool Decorator
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import tool



BORDER = "-"*65

###############################################################################
#   Function        :   multiply
#   Input Params    :   a,b
#   Output Params   :   None
#   Description     :   Multiply FUnction
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def multiply(a : int,b : int)->int:
    """Multiplies two numbers"""
    return a*b    
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    result = multiply.invoke({"a":8, "b":3})
    print(BORDER)
    print(f"Custom Tool-Multiply : {result}")
    print(BORDER)
    print(multiply.name)
    print(multiply.description)
    print(multiply.args)
    print(multiply.args_schema.model_json_schema())
    print(BORDER)   
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
    