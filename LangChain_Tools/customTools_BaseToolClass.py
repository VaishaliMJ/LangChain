"""----------------------------------------------------------------------------------
    Problem Statement   :   Cutom Tools LangChain :  Using Base Tool Class
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import BaseTool
from typing import Type
from pydantic import BaseModel,Field


BORDER = "-"*65
###############################################################################
#   Class           :   multiply Class
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Args Schema using Pydentic
#   Author          :   Vaishali M Jorwekar
###############################################################################
class multiplyClass(BaseModel):
    a : int =   Field(required=True,description="First Number to be added")
    b : int =   Field(required=True,description="Second Number to be added")
###############################################################################
#   Class           :   multiply tool class
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Search tool
#   Author          :   Vaishali M Jorwekar
###############################################################################
class multiplyToolClass(BaseTool):
    name :  str = "multiply"
    description : str = "Multiply two numbers"
    args_schema :   Type[BaseModel]   = multiplyClass
    
    def _run(self,a : int, b : int):
        return a * b
    
      

###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    multiplyTool = multiplyToolClass()
    result=multiplyTool.invoke({'a':9,'b':5})
    
    print(BORDER)
    print(f"Custom Tool - Base Tool Class : {result}")
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
    