import time
from qdrant_client import QdrantClient

class MemoryEngine:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection = collection_name

    def log_interaction(self, text_query, best_hit_id=None):
        """
        Updates the 'usage_count' for a specific fact in the database.
        This simulates 'Reinforcement Learning' - the more a fact is used, 
        the stronger it becomes in memory (visualized in the UI).
        """
        if best_hit_id is not None:
            try:
                # 1. Get current state of the record
                points = self.client.retrieve(
                    collection_name=self.collection,
                    ids=[best_hit_id]
                )
                
                if points:
                    point = points[0]
                    # Default to 0 if not present
                    current_count = point.payload.get("usage_count", 0)
                    new_count = current_count + 1
                    
                    # 2. Update the payload (Reinforcement)
                    self.client.set_payload(
                        collection_name=self.collection,
                        payload={
                            "usage_count": new_count, 
                            "last_accessed": int(time.time())
                        },
                        points=[best_hit_id]
                    )
                    print(f"🧠 Memory Reinforced: ID {best_hit_id} count -> {new_count}")
            except Exception as e:
                print(f"⚠️ Memory Update Failed: {e}")

    def get_stats(self):
        """
        Retrieves top 'remembered' facts for the UI visualization.
        """
        try:
            # Scroll through database to find items
            # In production, we would filter by 'usage_count > 0', 
            # but scrolling is faster for small datasets.
            response = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                with_payload=True
            )[0]
            
            # Sort by usage count (Highest first) to show most active memories
            active_memories = sorted(
                [p for p in response if p.payload.get("usage_count", 0) > 0],
                key=lambda x: x.payload.get("usage_count", 0),
                reverse=True
            )
            return active_memories
        except Exception as e:
            print(f"Memory Read Error: {e}")
            return []