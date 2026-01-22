🛡️ Misinformation Debunker
Misinformation Debunker is an advanced, agentic AI system designed to verify multimodal claims (text and images). It utilizes a Dual-Brain Architecture combined with a Neural Memory Engine to detect hoaxes with high precision. The system features an intelligent "Semantic Cross-Check" to filter false positives and learns from user interactions through reinforcement logic.

🚀 Key Features
🤖 Agentic Workflow: A modular system comprising three specialized agents:

Retriever Agent: Handles vector search (SBERT/CLIP) and applies memory boosts.

Forensic Agent: Performs zero-shot semantic verification to distinguish specific hoaxes from generic images (e.g., distinguishing a "Shark on Highway" hoax from a normal "Shark in Ocean").

Orchestrator: Manages the workflow, routing tasks and synthesizing the final verdict using Llama-3.

🧠 Neural Memory Engine: Implements reinforcement learning. Verified facts gain "experience points," allowing the system to recognize noisy, blurry, or cropped variations of known hoaxes over time.

👁️ Dual-Brain Retrieval: Uses SBERT for semantic text analysis and CLIP for dense visual vector matching.

📊 Transparent Analysis: A clean Streamlit interface that provides confidence scores, detailed evidence logs, and real-time visualization of the memory state.

🛠️ Tech Stack
Frontend: Streamlit

Vector Database: Qdrant (Local Embedded Mode)

AI Models:

Text Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)

Visual Embeddings: clip-ViT-B-32 (CLIP)

Reasoning Engine: Llama-3-70b (via Groq API)

📦 Installation Guide
1. Clone the Repository

Bash
git clone <your-repo-url>
cd Misinformation-Debunker
2. Set up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

MacOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate
Windows:

Bash
python -m venv venv
venv\Scripts\activate
3. Install Dependencies

Create a requirements.txt file (if not present) with the following content:

Plaintext
streamlit
qdrant-client
sentence-transformers
groq
python-dotenv
Pillow
pandas
torch
Then run:

Bash
pip install -r requirements.txt
4. Configure Environment Variables

Create a .env file in the root directory and add your Groq API key:

Ini, TOML
GROQ_API_KEY=gsk_your_actual_api_key_here
📂 Data Setup (Truth Database)
The system relies on a "Truth Database" of verified facts. Place your reference images and metadata in data/manual_db.

Directory Structure:

Plaintext
Misinformation-Debunker/
├── .env
├── app.py
├── requirements.txt
├── src/
│   ├── agent.py      <-- The 3 Agents (Retriever, Forensic, Orchestrator)
│   ├── memory.py     <-- Neural Memory Logic
│   └── ingestion.py  <-- Data Processing Script
└── data/
    └── manual_db/
        ├── images/         <-- Verified images (e.g., shark.jpg)
        └── metadata.csv    <-- Truth labels
Example metadata.csv Format:

Code snippet
filename,text,verdict,source
shark.jpg,A shark swimming on a flooded highway in Houston during Hurricane Harvey.,FAKE,Snopes
pope.jpg,Pope Francis wearing a white Balenciaga puffer jacket.,FAKE,Reuters
🏃‍♂️ How to Run
Step 1: Ingest Data (Build the Brain)

Run the ingestion script. This converts your images and text into vectors and generates augmented variations (crops, flips) to make the database robust.

Bash
python src/ingestion.py
Wait for the success message: 🎉 SUCCESS! Database expanded...

Step 2: Launch the Application

Start the Streamlit UI:

Bash
streamlit run app.py
The app will open automatically in your browser at http://localhost:8501.
