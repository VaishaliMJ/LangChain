"""----------------------------------------------------------------------------------
    Problem Statement   :   Built In Tools In LangChain
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool



BORDER = "-"*65
###############################################################################
#   Function        :   searchTool
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Search tool
#   Author          :   Vaishali M Jorwekar
###############################################################################
def searchTool():
    print(BORDER)
    print("Duck Duck Go Search Run")
    print(BORDER)
    searchTool = DuckDuckGoSearchRun()
    question = "List cultural Festivals in India"
    result = searchTool.invoke(question)
    print(BORDER)
    print(f"Result for '{question}'    \n{BORDER}")
    print(result.split("\n"))
    print(BORDER)
    
    print(f"Search Tool Name : {searchTool.name}")
    print(f"Search Tool Description : {searchTool.description}")
    print(f"Search Tool Args : {searchTool.args}")
    print(BORDER)
    print(BORDER)
###############################################################################
#   Function        :   shellTool
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Command Exceution tool
#   Author          :   Vaishali M Jorwekar
###############################################################################
def shellTool():
    print(BORDER)
    print("Shell Tool")
    print(BORDER)
    shellTool = ShellTool()
    result = shellTool.invoke("ls")
    print(BORDER)
    print(f"Result for 'ls'    \n{BORDER}")
    print(result.split("\n"))
    print(BORDER)
    
    print(f"Shell Tool Name : {shellTool.name}")
    print(f"Shell Tool Description : {shellTool.description}")
    print(f"Shell Tool Args : {shellTool.args}")
    print(BORDER)
    print(BORDER)
    
        
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    searchTool()
    shellTool()
    
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()
    