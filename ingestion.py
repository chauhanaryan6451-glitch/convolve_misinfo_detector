import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil

# --- CONFIG ---
BASE_DIR = "data/manual_db"
CSV_PATH = os.path.join(BASE_DIR, "metadata.csv")
IMG_DIR = os.path.join(BASE_DIR, "images")
DB_PATH = "data/truth_db"

def generate_augmentations(img_obj):
    """
    Creates 5 variations of the input image to make retrieval robust.
    Returns a list of (name_suffix, image_object).
    """
    variants = []
    
    # 1. Original
    variants.append(("orig", img_obj))
    
    # 2. Horizontal Flip (Catches mirrored reposts)
    variants.append(("flip", ImageOps.mirror(img_obj)))
    
    # 3. Grayscale (Catches B&W reposts)
    variants.append(("gray", ImageOps.grayscale(img_obj).convert("RGB")))
    
    # 4. Brightness Boost (Catches over-exposed photos)
    enhancer = ImageEnhance.Brightness(img_obj)
    variants.append(("bright", enhancer.enhance(1.5)))
    
    # 5. Center Crop (Catches zoomed-in/cropped versions)
    width, height = img_obj.size
    # Crop 10% from all sides
    left = width * 0.1
    top = height * 0.1
    right = width * 0.9
    bottom = height * 0.9
    variants.append(("crop", img_obj.crop((left, top, right, bottom))))
    
    return variants

def ingest_augmented_data():
    print("🧬 Initializing Augmented Ingestion...")
    
    # 1. CLEAN SLATE
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    client = QdrantClient(path=DB_PATH)
    
    print("   ↳ Loading AI Brains...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    vision_model = SentenceTransformer("clip-ViT-B-32")

    # 2. SETUP VECTOR CONFIG
    client.recreate_collection(
        collection_name="multimodal_knowledge",
        vectors_config={
            "text_vec": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "image_vec": models.VectorParams(size=512, distance=models.Distance.COSINE),
        }
    )

    df = pd.read_csv(CSV_PATH)
    print(f"   🚀 Expanding {len(df)} base records...")
    
    total_vectors = 0
    points = []
    
    for index, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join(IMG_DIR, filename)
        
        try:
            base_img = Image.open(img_path).convert("RGB")
            
            # --- GENERATE VARIANTS ---
            variants = generate_augmentations(base_img)
            
            # The Text Vector is the same for all variants
            txt_emb = text_model.encode(row['text']).tolist()
            
            for i, (suffix, img_variant) in enumerate(variants):
                # Unique Image Vector for this variation
                img_emb = vision_model.encode(img_variant).tolist()
                
                payload = {
                    "text": row['text'],
                    "verdict": row['verdict'],
                    "source": row['source'],
                    "image_path": img_path,
                    "augmentation": suffix  # Track which version matched
                }

                # Create ID: index * 100 + variation_index 
                # e.g., Shark (ID 0) -> 0, 1, 2, 3, 4
                point_id = (index * 100) + i
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector={
                        "text_vec": txt_emb,
                        "image_vec": img_emb
                    },
                    payload=payload
                ))
                total_vectors += 1
                
            print(f"   ✅ Indexed {filename} (+5 variations)")

        except Exception as e:
            print(f"   ❌ Error {filename}: {e}")

    # 3. UPLOAD BATCH
    client.upsert(collection_name="multimodal_knowledge", points=points)
    print(f"\n🎉 SUCCESS! Database expanded to {total_vectors} robust vectors.")
    print("   Now the system can recognize crops, flips, and edits!")

if __name__ == "__main__":
    ingest_augmented_data()