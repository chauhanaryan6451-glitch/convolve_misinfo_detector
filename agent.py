import os
import warnings
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from src.memory import MemoryEngine
import torch.nn.functional as F
import torch

warnings.filterwarnings("ignore")
load_dotenv()

# --- AGENT 1: THE RETRIEVER (Search & Memory) ---
class RetrieverAgent:
    def __init__(self, client, collection_name):
        print("   ↳ Initializing Retriever Agent...")
        self.client = client
        self.collection = collection_name
        # Load Embedding Models
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vision_model = SentenceTransformer("clip-ViT-B-32")

    def retrieve(self, query_input, input_type="text"):
        """
        Executes vector search and applies MEMORY BOOST logic.
        """
        # 1. Encode
        if input_type == "text":
            vector = self.text_model.encode(query_input).tolist()
            vector_name = "text_vec"
        else:
            vector = self.vision_model.encode(query_input).tolist()
            vector_name = "image_vec"

        # 2. Search
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=models.NamedVector(name=vector_name, vector=vector),
            limit=5
        )

        # 3. Apply Memory Reinforcement (The "Learning" Logic)
        for h in hits:
            usage = h.payload.get('usage_count', 0)
            # Logic: Cap boost at 0.10 (10%)
            boost = min(usage * 0.01, 0.10)
            if boost > 0:
                h.score += boost
        
        # Re-rank after boosting
        hits = sorted(hits, key=lambda x: x.score, reverse=True)
        return hits

# --- AGENT 2: THE FORENSIC ANALYST (Visual Logic) ---
class ForensicAgent:
    def __init__(self, vision_model):
        print("   ↳ Initializing Forensic Agent...")
        self.vision_model = vision_model

    def semantic_cross_check(self, image_obj, claim_text):
        """
        Performs the Zero-Shot check: 'Generic Context' vs 'Hoax Context'.
        """
        control_text = "A normal, generic photo of this object in its natural habitat."
        
        # Encode inputs to Tensor
        img_emb = self.vision_model.encode(image_obj, convert_to_tensor=True)
        text_claim_emb = self.vision_model.encode(claim_text, convert_to_tensor=True)
        text_control_emb = self.vision_model.encode(control_text, convert_to_tensor=True)
        
        # Calculate Cosine Similarity
        score_claim = torch.cosine_similarity(img_emb, text_claim_emb, dim=0).item()
        score_control = torch.cosine_similarity(img_emb, text_control_emb, dim=0).item()
        
        # Logic: If Generic > Specific, it's a False Positive.
        if score_control > score_claim:
            return False, f"Image matches 'Generic Context' ({score_control:.2f}) better than 'Hoax Context' ({score_claim:.2f})."
        
        return True, "Image aligns with Hoax Context."

# --- AGENT 3: THE ORCHESTRATOR (Coordinator) ---
class Orchestrator:
    def __init__(self):
        print("🤖 System Orchestrator Initializing...")
        self.client = QdrantClient(path="data/truth_db")
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.collection = "multimodal_knowledge"
        
        # Initialize Sub-Agents
        self.memory = MemoryEngine(self.client, self.collection)
        self.retriever = RetrieverAgent(self.client, self.collection)
        self.forensic = ForensicAgent(self.retriever.vision_model) # Share model to save RAM

    def process_request(self, user_query, input_type="text", image_obj=None):
        # 1. DELEGATE TO RETRIEVER
        hits = self.retriever.retrieve(user_query if input_type=="text" else image_obj, input_type)
        
        if not hits:
            return "UNVERIFIED", "No matches found.", 0.0

        top_hit = hits[0]
        score = top_hit.score
        meta = top_hit.payload
        base_verdict = meta.get('verdict', 'UNKNOWN')

        # --- LOGIC GATES (Preserved) ---
        IMG_FLOOR = 0.82      
        TXT_FLOOR = 0.40      
        threshold = TXT_FLOOR if input_type == "text" else IMG_FLOOR

        # Gate 1: Baseline Threshold
        if score < threshold:
             if input_type == "text":
                # Keyword Rescue Logic
                q_words = set(user_query.lower().split())
                r_words = set(meta['text'].lower().split())
                common = q_words.intersection(r_words) - {"the", "a", "is", "of", "in"}
                if len(common) == 0:
                    return "UNVERIFIED", f"⚠️ Low Similarity ({score:.2f}) and no keyword overlap.", 0.0
             else:
                return "UNVERIFIED", f"⚠️ Similarity ({score:.2f}) below verification threshold.", 0.0

        # Gate 2: DELEGATE TO FORENSIC AGENT (Images Only)
        if input_type == "image" and image_obj:
            passed, reason = self.forensic.semantic_cross_check(image_obj, meta['text'])
            if not passed:
                return "UNVERIFIED", f"⚠️ **False Positive Detected:** {reason} (Vector Score: {score:.2f})", 0.0

        # Gate 3: LLM Synthesis & Memory Update
        instruction = f"""
        The user's input matches verified record: '{meta['text']}'.
        Identity Match Score (Boosted): {score:.3f}.
        Semantic Check: PASSED.
        VERDICT: {base_verdict}.
        """
        
        # Write to Memory (Reinforcement)
        self.memory.log_interaction(user_query, best_hit_id=top_hit.id)

        prompt = f"""
        Role: Senior Forensic Analyst.
        EVIDENCE:
        - Claim: {meta['text']}
        - Verdict: {base_verdict}
        - Source: {meta['source']}
        - System Note: {instruction}
        
        OUTPUT FORMAT:
        ## 🚨 Verdict: [VERDICT]
        **Analysis:** [Confirm match.]
        **Source:** [Source Name]
        """
        
        confidence = min(max((score - 0.5) * 2.0, 0.1), 0.99)
        
        try:
            res = self.groq.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0
            )
            return res.choices[0].message.content, f"Match: {meta['text']}\nSource: {meta['source']}", confidence
        except Exception as e:
            return "Error", str(e), 0.0