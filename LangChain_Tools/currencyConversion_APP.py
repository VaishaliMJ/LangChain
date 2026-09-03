"""----------------------------------------------------------------------------------
    Problem Statement   :   Real Time Currency Conversion Streamlit APP
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
import streamlit as st

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
#   Function        :   getCurrencyCodes
#   Input Params    :   None
#   Output Params   :   Currency codes
#   Description     :   Returns currency codes 
#   Author          :   Vaishali M Jorwekar
###############################################################################
def getCurrencyCodes():
    currencyCodes = [
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
    "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
    "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD",
    "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG", "HUF", "IDR", "ILS",
    "INR", "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR",
    "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD",
    "LSL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU",
    "MUR", "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK",
    "NPR", "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG",
    "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK",
    "SGD", "SHP", "SLE", "SLL", "SOS", "SRD", "STN", "SVC", "SYP", "SZL",
    "THB", "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH",
    "UGX", "USD", "UYU", "UZS", "VED", "VES", "VND", "VUV", "WST", "XAF",
    "XCD", "XOF", "XPF", "YER", "ZAR", "ZMW", "ZWG"
    ]
    return currencyCodes

###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    load_dotenv()
    
    #################################################################
    
    st.title("Currency Conversion")
    currencyInput=st.text_input("Enter Currency for conversion")
    currencyCodes=getCurrencyCodes()
    
    baseCurrency = st.selectbox("Select Base Currency", options=currencyCodes)
    targetCurrency = st.selectbox("Select Target Currency", options=currencyCodes)
    
    if st.button("Convert Currency"):
        if baseCurrency != targetCurrency:
            
        #################################################################
            
            llmModel=ChatOllama(model="llama3.1", temperature=0)
            
            llmWithTools=llmModel.bind_tools([getConversionFactor,convertCurrency])
            
            messages=[HumanMessage(f"What is conversion factor between {baseCurrency} and {targetCurrency},and based on that  can you convert {currencyInput} {baseCurrency} to {targetCurrency}")]
            
            
            aiMessage = llmWithTools.invoke(messages)
            
            loop_trigger = "Execute first tool and get the value of conversion rate"
            if aiMessage.content and loop_trigger in aiMessage.content:
                aiMessage.content = aiMessage.content.replace(loop_trigger, "").strip()


            messages.append(aiMessage)
            print(f"AI Messages : {aiMessage}")
            if aiMessage.tool_calls:
                
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
                    finalResponse = response.replace(loop_trigger,"").strip() 
                    st.write(finalResponse)
                    print(BORDER)
                    st.stop()
                  
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()