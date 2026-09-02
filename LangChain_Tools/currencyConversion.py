"""----------------------------------------------------------------------------------
    Problem Statement   :   Real Time Currency Conversion
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import tool
from langchain_ollama import ChatOllama
from typing import Annotated
from langchain_core.messages import HumanMessage
import requests
import os
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
import json


URL_CURRENCY_API=f"https://v6.exchangerate-api.com/v6/"
BORDER="-"*65

###############################################################################
#   Function        :   getConversionFactor
#   Input Params    :   baseCurrency,targetCurrency
#   Output Params   :   Real time conversion factor
#   Description     :   Finds Real time conversion factor for currency 
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def getConversionFactor(baseCurrency:str,targetCurrency:str)->float:
    """
    This Function fectches the conversion factor between a given base currency and a target currency

    Args:
        baseCurrency (_type_): _description_
        targetCurrency (_type_): _description_

    Returns:
        float: _description_
    """
    API_KEY=os.getenv("EXCHANGE_RATE_API")
    #https://v6.exchangerate-api.com/v6/YOUR-API-KEY/pair/EUR/GBP

    url=f'{URL_CURRENCY_API}/{API_KEY}/pair/{baseCurrency}/{targetCurrency}'
    response=requests.get(url)
    
    return response.json()
    
###############################################################################
#   Function        :   convertCurrency
#   Input Params    :   baseCurrencyValue,conversionFactor
#   Output Params   :   Real time conversion factor
#   Description     :   Converts currency into 
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def convertCurrency(baseCurrencyValue:float,conversionFactor:Annotated[float,InjectedToolArg])->float:    
    """
        Given a currency conversion rate this function 
        calculates the target currency value from a given base currency value

    Args:
        baseCurrencyValue (float): _description_
        conversionFactor (float): _description_

    Returns:
        float: _description_
    """
    return baseCurrencyValue * conversionFactor
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    load_dotenv()
    response=getConversionFactor.invoke({'baseCurrency':'USD',
                                'targetCurrency':'INR'})
    
    conversionFactor=response['conversion_rate']
    print(f"Input Currency Value    :   20")
    print(f"Conversion Factor Rate  : {conversionFactor}")
    
    
    convertedCurrency=convertCurrency.invoke({
        'baseCurrencyValue':20.0,
        'conversionFactor': conversionFactor
    })
    
    print(f"Converted Currency  :   {convertedCurrency}")
    
    llmModel=ChatOllama(model="llama3.1", temperature=0)
    
    llmWithTools=llmModel.bind_tools([getConversionFactor,convertCurrency])
    
    messages=[HumanMessage("What is conversion factor between USD and INR,and based on that  can you convert 20 USD to INR")]
    
    aiMessage = llmWithTools.invoke(messages)
    messages.append(aiMessage)
    
    print(f"AI Messages : {aiMessage}")
    for toolCall in aiMessage.tool_calls:
        """ Execute first tool and get the value of conversion rate """
        if toolCall['name']=='getConversionFactor':
            toolMessage_1=getConversionFactor.invoke(toolCall)
           
            conversionRate= json.loads(toolMessage_1.content)['conversion_rate']
            messages.append(toolMessage_1)
            
        if toolCall['name']=='convertCurrency':
            toolCall['args']['conversionFactor']=conversionRate
            
            toolMessage_2=convertCurrency.invoke(toolCall)
            messages.append(toolMessage_2)   
            
        
        print("\n\n\n")
        print(BORDER)
        print(f"Messages    :   {messages}")
        print(BORDER)
        
        response=llmWithTools.invoke(messages).content
        print(BORDER)
        print(response)
        print(BORDER)
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()