import os
import psutil
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
    Route to: MONITORING_AGENT (for CPU/RAM), TASK_AGENT (for chat/code/name), or FINISH.
    Reply ONLY with the uppercase name of the agent.
    """
    response = llm.invoke(prompt)
    decision = response.content.upper()
    if "MONITORING" in decision: return {"next_agent": "MONITORING_AGENT"}
    if "TASK" in decision: return {"next_agent": "TASK_AGENT"}
    return {"next_agent": "FINISH"}

def monitoring_agent(state: AgentState):
    print("\n[System] Fetching metrics...")
    cpu, ram = psutil.cpu_percent(interval=0.5), psutil.virtual_memory().percent
    return {"messages": [AIMessage(content=f"Monitoring Agent: Your CPU is at {cpu}% and RAM is {ram}%.")]}

def task_agent(state: AgentState):
    print("\n[System] Task Agent thinking...")
    response = llm.invoke(state['messages'])
    
    # Save code to file if detected
    if "```python" in response.content or "import " in response.content:
        with open("generated_code.py", "w") as f:
            f.write(response.content)
        print("!!! Saved to generated_code.py !!!")

    return {"messages": [AIMessage(content=f"Task Agent: {response.content}")]}

# --- STEP 4: BUILD THE GRAPH (Logic Only) ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_router)
workflow.add_node("monitoring_agent", monitoring_agent)
workflow.add_node("task_agent", task_agent)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda x: x["next_agent"], 
                              {"MONITORING_AGENT": "monitoring_agent", "TASK_AGENT": "task_agent", "FINISH": END})
workflow.add_edge("monitoring_agent", "supervisor")
workflow.add_edge("task_agent", "supervisor")

# We DO NOT compile it with memory here anymore to avoid the error
app = workflow.compile() 

if __name__ == "__main__":
    # For Terminal Use: Create the connection here
    with SqliteSaver.from_conn_string("cloud_history.db") as saver:
        terminal_app = workflow.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "terminal_user"}}
        print("--- SmartCloud Terminal Active ---")
        while True:
            u_input = input("\nYou: ")
            if u_input.lower() in ["exit", "quit"]: break
            for output in terminal_app.stream({"messages": [HumanMessage(content=u_input)]}, config=config):
                for key, value in output.items():
                    if 'messages' in value:
                        print(f"Agent: {value['messages'][-1].content}")