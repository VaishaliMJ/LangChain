"""----------------------------------------------------------------------------------
    Problem Statement   :   Cutom Tools LangChain :  Using StructuredTool
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import StructuredTool
from pydantic import BaseModel,Field


BORDER = "-"*65
###############################################################################
#   Class           :   multiply
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Search tool
#   Author          :   Vaishali M Jorwekar
###############################################################################
class multiplyClass(BaseModel):
    a : int =   Field(required=True,description="First Number to be added")
    b : int =   Field(required=True,description="Second Number to be added")
    
###############################################################################
#   Function        :   multiplyFunc
#   Input Params    :   a,b
#   Output Params   :   None
#   Description     :   Multiply FUnction
#   Author          :   Vaishali M Jorwekar
###############################################################################
def multiplyFunc(a : int,b : int):
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
    multiplyTool = StructuredTool.from_function(
        
            func=multiplyFunc,
            name="multiply",
            description="Multiply two numbers",
            args_schema=multiplyClass
        
    )
    result=multiplyTool.invoke({'a':9,'b':5})
    
    print(BORDER)
    print(f"Custom Tool-StructuredTool : {result}")
    print(BORDER)
    
    print(multiplyTool.name)
    print(multiplyTool.description)
    print(multiplyTool.args)
    print(multiplyTool.args_schema.model_json_schema())
    print(BORDER)   
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
    