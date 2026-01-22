import streamlit as st
from PIL import Image
import time
from src.agent import Orchestrator

# --- CONFIGURATION ---
st.set_page_config(page_title="Misinformation Debunker", layout="wide", page_icon="🛡️")

# --- INITIALIZE SYSTEM ---
@st.cache_resource
def get_system():
    return Orchestrator()

try:
    system = get_system()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CSS ---
st.markdown("""
<style>
    .warning-box { padding: 1rem; background-color: #3d3d0e; border-left: 5px solid #ffa500; border-radius: 5px; margin-bottom: 1rem; color: #e0e0e0; }
    .warning-title { font-weight: bold; color: #ffa500; font-size: 1.1rem; margin-bottom: 0.5rem; }
    .high-conf { color: #21c354; font-weight: bold; }
    .low-conf { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Neural Memory State")
    st.markdown("Real-time visualization of the system's learning process.")
    
    if st.button("🔄 Refresh Stats"):
        try:
            memories = system.memory.get_stats()
            total_xp = sum([m.payload.get('usage_count', 0) for m in memories])
            st.metric("Total Experience Points", total_xp)

            if memories:
                active = [m for m in memories if m.payload.get('usage_count', 0) > 0]
                if not active: st.caption("System is currently Tabula Rasa.")
                for m in active[:5]:
                    usage = m.payload.get('usage_count', 0)
                    bar = "🟦" * usage if usage < 5 else "🟪" * usage
                    st.caption(f"**{m.payload['text'][:25]}...**\n{bar} ({usage})")
        except:
            st.warning("Memory Offline")

    st.divider()
    
    st.header("📸 Visual Input")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Analysis Target", use_container_width=True)
        
        if st.button("🚀 Run Agentic Scan", type="primary"):
            st.session_state.messages.append({"role": "user", "content": "🖼️ [Image Scan Request]"})
            with st.spinner("Orchestrating Agents..."):
                response, evidence, conf = system.process_request("", input_type="image", image_obj=image)
                if "UNVERIFIED" in response or conf == 0.0: conf = 0.0
                st.session_state.messages.append({"role": "assistant", "content": response, "evidence": evidence, "confidence": conf})
                st.rerun()

    st.divider()
    st.caption("System Status: 🟢 Online (Agentic Mode)")

# --- MAIN UI ---
st.title("🛡️ Misinformation Debunker")
st.markdown("Multi-Agent AI System for Multimodal Verification")

# 1. DISPLAY HISTORY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "confidence" in msg and msg['confidence'] == 0.0:
            st.markdown(f"<div class='warning-box'><div class='warning-title'>⚠️ AGENT UNVERIFIED</div>{msg.get('evidence', '')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])
            if "evidence" in msg and msg['confidence'] > 0:
                color = "high-conf" if msg['confidence'] > 0.8 else "low-conf"
                st.markdown(f"Confidence: <span class='{color}'>{msg['confidence']*100:.0f}%</span>", unsafe_allow_html=True)
                with st.expander("Agent Findings"): st.text(msg["evidence"])

# 2. TEXT INPUT
if prompt := st.chat_input("Enter a claim..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Orchestrator assigning tasks..."):
            response, evidence, conf = system.process_request(prompt, input_type="text")
            if "UNVERIFIED" in response or conf == 0.0:
                    st.markdown(f"<div class='warning-box'><div class='warning-title'>⚠️ AGENT UNVERIFIED</div>{evidence}</div>", unsafe_allow_html=True)
            else:
                st.markdown(response)
                st.markdown(f"**Confidence:** {conf*100:.0f}%")
                with st.expander("Agent Findings"): st.text(evidence)

    st.session_state.messages.append({"role": "assistant", "content": response, "evidence": evidence, "confidence": conf})