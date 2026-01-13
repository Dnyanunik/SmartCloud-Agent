import streamlit as st
import psutil
from agent_brain import workflow # Only import the workflow
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="SmartCloud AI", page_icon="☁️", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🖥️ System Monitor")
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    st.metric("CPU Usage", f"{cpu}%")
    st.progress(cpu/100)
    st.metric("RAM Usage", f"{ram}%")
    st.progress(ram/100)

st.title("☁️ SmartCloud Agent Dashboard")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # CREATE THE SAVER HERE TO AVOID THE ATTRIBUTE ERROR
        with SqliteSaver.from_conn_string("cloud_history.db") as saver:
            web_app = workflow.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "web_user_1"}}
            
            final_text = ""
            # Run the agent brain
            for output in web_app.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                for key, value in output.items():
                    if 'messages' in value:
                        final_text = value['messages'][-1].content
            
            st.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})