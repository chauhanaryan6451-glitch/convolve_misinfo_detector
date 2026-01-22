# 🛡️ Misinformation Debunker

**Misinformation Debunker** is an advanced, agentic AI system designed to verify multimodal claims (text and images). It utilizes a **Dual-Brain Architecture** combined with a **Neural Memory Engine** to detect hoaxes with high precision.

---

* **🤖 Agentic Workflow:**
  * **Retriever Agent:** Handles vector search and memory boosting.
  * **Forensic Agent:** Performs zero-shot semantic verification (distinguishes specific hoaxes from generic images).
  * **Orchestrator:** Manages the workflow and synthesizes the final verdict.
* **🧠 Neural Memory Engine:** Implements reinforcement learning. Verified facts gain "experience points," allowing the system to recognize noisy inputs.
* **👁️ Dual-Brain Retrieval:** Uses SBERT (Text) and CLIP (Vision).

---

## 📂 Project Structure

Ensure your project files are organized like this:

```text
FACT_CHECKER/
├── .env                     # API keys
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── data/
│   ├── manual_db/
│   │   ├── images/          # Reference images (e.g., shark.jpg)
│   │   └── metadata.csv     # Metadata for manual database
│   ├── truth_db/            # Vector database (Qdrant collection)
│   ├── test.jpeg            # Test image
│   └── test2.jpeg           # Test image
└── src/
    ├── agent.py             # Agent logic
    ├── ingestion.py         # Data processing script
    └── memory.py            # Neural memory engine

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Database:** Qdrant (Local Embedded)
* **AI Models:** SBERT, CLIP, Llama-3-70b (Groq)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd convolve_misinfo_detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory and add your Groq key:
```ini
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the Application

**Step 1: Ingest Data**
Process images and build the vector database.
```bash
python src/ingestion.py
```

**Step 2: Launch App**
Start the interface.
```bash
streamlit run app.py
```
