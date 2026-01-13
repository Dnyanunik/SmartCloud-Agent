import os
import psutil
import subprocess
import operator
from typing import TypedDict, Annotated, List
from langchain_groq import ChatGroq 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from dotenv import load_dotenv

load_dotenv()
# --- CONFIG ---
MY_GROQ_KEY = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=MY_GROQ_KEY, temperature=0.1)

# --- NODES ---
def supervisor_router(state: AgentState):
    messages = state['messages']
    if isinstance(messages[-1], AIMessage):
        return {"next_agent": "FINISH"}

    prompt = f"""
    You are the SmartCloud Supervisor. 
    CONVERSATION HISTORY: {messages[-5:]} 
    
    Decision Logic:
    1. If user asks for hardware/CPU/RAM stats -> MONITORING_AGENT
    2. If user asks for code, name, or general help -> TASK_AGENT
    3. If the request is already answered -> FINISH
    
    Reply ONLY with: MONITORING_AGENT, TASK_AGENT, or FINISH.
    """
    response = llm.invoke(prompt)
    decision = response.content.upper()
    if "MONITORING" in decision: return {"next_agent": "MONITORING_AGENT"}
    if "TASK" in decision: return {"next_agent": "TASK_AGENT"}
    return {"next_agent": "FINISH"}

def monitoring_agent(state: AgentState):
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    # Phase 4: Logic for proactive alerts
    alert = ""
    if ram > 90: alert = "\n⚠️ ALERT: Memory usage is critically high!"
    
    report = f"Monitoring Agent: CPU is {cpu}% and RAM is {ram}%.{alert}"
    return {"messages": [AIMessage(content=report)]}

def task_agent(state: AgentState):
    response = llm.invoke(state['messages'])
    # Action: File Saving
    if "```python" in response.content:
        with open("generated_code.py", "w") as f:
            f.write(response.content)
    return {"messages": [AIMessage(content=f"Task Agent: {response.content}")]}

def action_agent(state: AgentState):
    """A unique node that can actually execute system-level commands."""
    last_msg = state['messages'][-1].content
    
    # The LLM decides what command is needed based on the problem
    prompt = f"The system has an issue: {last_msg}. Provide a one-line PowerShell command to investigate or fix it. Reply ONLY with the command."
    response = llm.invoke(prompt)
    command = response.content.strip()
    
    # We don't execute automatically for safety, we show it to the user
    return {"messages": [AIMessage(content=f"🛠️ AUTO-FIX SUGGESTED: I recommend running this command: `{command}`. Would you like me to execute this for you?")]}

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_router)
workflow.add_node("monitoring_agent", monitoring_agent)
workflow.add_node("task_agent", task_agent)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda x: x["next_agent"], 
                              {"MONITORING_AGENT": "monitoring_agent", "TASK_AGENT": "task_agent", "FINISH": END})
workflow.add_edge("monitoring_agent", "supervisor")
workflow.add_edge("task_agent", "supervisor")

app = workflow.compile()