import streamlit as st
import psutil
import pandas as pd
import time
from agent_brain import workflow
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from fpdf import FPDF
import datetime

# 1. MUST BE FIRST: Page Configuration
st.set_page_config(page_title="SmartCloud Agent", page_icon="☁️", layout="wide")

# 2. Matrix Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }
    [data-testid="stMetricValue"] {
        color: #00ff41;
    }
    </style>
    """, unsafe_allow_html=True)

# --- REPORT GENERATION FUNCTION ---
def generate_system_report(cpu, ram, messages):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "SmartCloud System Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # System Metrics Section
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(0, 10, "1. Real-Time Hardware Analysis", ln=True, fill=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Total CPU Usage: {cpu}%", ln=True)
    pdf.cell(0, 10, f"Total RAM Usage: {ram}%", ln=True)
    pdf.ln(5)

    # Conversation Log Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "2. Agent Interaction Log", ln=True, fill=True)
    pdf.set_font("Arial", "", 10)
    for msg in messages:
        role = "User" if msg["role"] == "user" else "AI Agent"
        clean_text = msg["content"].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, f"{role}: {clean_text}")
        pdf.ln(2)

    return bytes(pdf.output())

# --- INITIALIZE HISTORY ---
if "cpu_history" not in st.session_state:
    st.session_state.cpu_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: LIVE MONITORING ---
with st.sidebar:
    st.title("🛡️ Cloud Monitor")
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    
    st.session_state.cpu_history.append(cpu)
    if len(st.session_state.cpu_history) > 20:
        st.session_state.cpu_history.pop(0)
    
    st.metric("CPU Usage", f"{cpu}%")
    st.progress(cpu/100)
    st.metric("RAM Usage", f"{ram}%")
    st.progress(ram/100)
    
    st.subheader("CPU Trend")
    st.line_chart(st.session_state.cpu_history)
    
    # Report Download Section
    st.divider()
    st.subheader("📄 Reporting")
    if st.button("Generate Final Analysis"):
        pdf_bytes = generate_system_report(cpu, ram, st.session_state.messages)
        st.download_button(
            label="📩 Download PDF Report",
            data=pdf_bytes,
            file_name=f"SmartCloud_Report_{datetime.date.today()}.pdf",
            mime="application/pdf"
        )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("☁️ SmartCloud Multi-Agent Dashboard")
st.caption("Phase 4: Proactive Autonomous Management Active")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("How can I assist your business today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We wrap the generator in a list to consume it
        with SqliteSaver.from_conn_string("cloud_history.db") as saver:
            web_app = workflow.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "main_user_1"}} # Unique ID per session
            
            final_res = ""
            for output in web_app.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                for key, value in output.items():
                    if 'messages' in value:
                        final_res = value['messages'][-1].content
            
            st.markdown(final_res)
            st.session_state.messages.append({"role": "assistant", "content": final_res})