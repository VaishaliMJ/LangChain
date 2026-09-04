"""----------------------------------------------------------------------------------
    Problem Statement   :   Based on query find the weather of a city(AI Agent)
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
import streamlit as st
import os,requests

#from langchain_classic import hub
#from langchainhub import hub
from langsmith import Client
from langchain_classic.agents import create_react_agent, AgentExecutor

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

WEATHER_URL="http://api.weatherstack.com/current?"

###############################################################################
#   Function        :   loadSearchTool
#   Input Params    :   None
#   Output Params   :   Search Tool 
#   Description     :   Loads Search Tool
#   Author          :   Vaishali M Jorwekar
###############################################################################
def loadSearchTool():
    return DuckDuckGoSearchRun()
###############################################################################
#   Function        :   getWeatherTool
#   Input Params    :   None
#   Output Params   :   Weather tool
#   Description     :   Returns weather tool
#   Author          :   Vaishali M Jorwekar
###############################################################################
@tool
def getWeatherTool(city:str)->str:
    """
    This Function feches the current weather data for given city

    Args:
        city (str): Name of the city 

    Returns:
        str: Returns city current weather
    """
    API_KEY=os.getenv("WEATHER_API_KEY")
    url=f"{WEATHER_URL}/access_key={API_KEY}&query={city}"
    response=requests.get(url) 
    return response.json()
###############################################################################
#   Function        :   loadLLMModel
#   Input Params    :   None
#   Output Params   :   LLM Model
#   Description     :   Fetch llm model
#   Author          :   Vaishali M Jorwekar
###############################################################################
def loadLLMModel():
    model=ChatOllama(
        model="llama3.1",
        temperature=0.0
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
    load_dotenv()
    
    
    llmModel=loadLLMModel()
    searchTool=loadSearchTool()
    #os.environ["LANGCHAIN_DANGEROUSLY_PULL_PUBLIC_PROMPT"] = "true"
    client=Client()
    prompt=client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)
    
    #Create agent
    agent=create_react_agent(
        llm=llmModel,
        tools=[searchTool,getWeatherTool],
        prompt=prompt
    )
    
    #Agent Executor
    agentExecutor=AgentExecutor(
        agent=agent,
        tools=[searchTool,getWeatherTool],
        verbose=True
    )
    
    response=agentExecutor.invoke({"input":"Find the captial of Maharashtra,then find it's current weather condition"}) 
    print(response)
    print("\n\n\n\n")
    print(response['output'])
##############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()