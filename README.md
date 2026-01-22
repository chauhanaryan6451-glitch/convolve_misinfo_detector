# 🛡️ Misinformation Debunker

**Misinformation Debunker** is an advanced, agentic AI system designed to verify multimodal claims (text and images). It utilizes a **Dual-Brain Architecture** combined with a **Neural Memory Engine** to detect hoaxes with high precision.

---

## 🚀 Key Features

* **🤖 Agentic Workflow:**
  * **Retriever Agent:** Handles vector search and memory boosting.
  * **Forensic Agent:** Performs zero-shot semantic verification (distinguishes specific hoaxes from generic images).
  * **Orchestrator:** Manages the workflow and synthesizes the final verdict.
* **🧠 Neural Memory Engine:** Implements reinforcement learning. Verified facts gain "experience points," allowing the system to recognize noisy inputs.
* **👁️ Dual-Brain Retrieval:** Uses SBERT (Text) and CLIP (Vision).

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Database:** Qdrant (Local Embedded)
* **AI Models:** SBERT, CLIP, Llama-3-70b (Groq)

---

## 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
Install Dependencies

Bash
pip install -r requirements.txt
Configure API Key Create a .env file:

Ini, TOML
GROQ_API_KEY=gsk_your_actual_key_here
🏃‍♂️ How to Run
Step 1: Ingest Data Process images and build the vector database.

Bash
python src/ingestion.py
Step 2: Launch App Start the interface.

Bash
streamlit run app.py
