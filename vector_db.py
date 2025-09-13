from pymongo import MongoClient
import chromadb
from qdrant_client import QdrantClient
from supabase import create_client, Client
from dotenv import load_dotenv
from qdrant_client import models as qdrant_models
load_dotenv()
import os
import numpy as np
import pickle
import json

# qdrant < mongodb < chromadb (supabase cần tạo bảng thủ công)

# Các em có thể tự thêm vector database mới hoặc dùng các database có sẵn
class VectorDatabase:
    def __init__(self, db_type: str):
        self.db_type = db_type
        if self.db_type == "mongodb":
            self.client = MongoClient(os.getenv("MONGODB_URI"))
        elif self.db_type == "chromadb":
            self.client = chromadb.Client()
        elif self.db_type == "qdrant":
            self.client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY"))
        elif self.db_type == "supabase":
            url: str = os.environ.get("SUPABASE_URL")
            key: str = os.environ.get("SUPABASE_KEY")
            supabase: Client = create_client(
                supabase_url=url,
                supabase_key=key
                )
            self.client = supabase
    def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists for Qdrant, create if it doesn't"""
        if self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Info] Collection '{collection_name}' not found. Creating it...")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=1024,  # BGE-M3 ColBERT dimension
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                
                # Create index for title field to enable filtering
                print(f"[Info] Creating index for 'title' field...")
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="title",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
                return True  # Collection was created
        return False  # Collection already existed or not Qdrant
    def insert_document(self, collection_name: str, document: dict):
        # Handle ColBERT embeddings (multi-vector format)
        embedding = document["embedding"]
        is_colbert = isinstance(embedding, np.ndarray) and embedding.ndim == 2
        
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            
            if is_colbert:
                doc = document.copy()
                doc["embedding"] = pickle.dumps(embedding)
                doc["is_colbert"] = True
            else:
                doc = document.copy()
                doc["is_colbert"] = False
            collection.insert_one(doc)
            
        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name=collection_name)
            
            if is_colbert:
                # For ColBERT, store the serialized multi-vector in metadata
                # Use mean pooling as the main vector for ChromaDB compatibility
                mean_embedding = np.mean(embedding, axis=0).tolist()
                collection.add(
                    documents=[document["information"]],
                    embeddings=[mean_embedding],
                    ids=[document["title"]],
                    metadatas=[{
                        "colbert_embedding": pickle.dumps(embedding).hex(),
                        "is_colbert": True,
                        "embedding_shape_0": int(embedding.shape[0]),
                        "embedding_shape_1": int(embedding.shape[1])
                    }]
                )
            else:
                collection.add(
                    documents=[document["information"]],
                    embeddings=[document["embedding"]],
                    ids=[document["title"]],
                    metadatas=[{"is_colbert": False}]
                )
                
        elif self.db_type == "qdrant":
            self._ensure_collection_exists(collection_name)
            
            if is_colbert:
                # Store mean pooled vector and ColBERT data in payload
                mean_embedding = np.mean(embedding, axis=0).tolist()
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            "id": hash(document["title"]) % (2**63),
                            "vector": mean_embedding,
                            "payload": {
                                "title": document["title"],
                                "information": document["information"],
                                "colbert_embedding": pickle.dumps(embedding).hex(),
                                "is_colbert": True,
                                "embedding_shape_0": int(embedding.shape[0]),
                                "embedding_shape_1": int(embedding.shape[1])
                            }
                        }
                    ]
                )
            else:
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            "id": hash(document["title"]) % (2**63),
                            "vector": document["embedding"],
                            "payload": {
                                "title": document["title"],
                                "information": document["information"],
                                "is_colbert": False
                            }
                        }
                    ]
                )
                
        elif self.db_type == "supabase":
            if is_colbert:
                document_copy = document.copy()
                document_copy["embedding"] = pickle.dumps(embedding).hex()
                document_copy["is_colbert"] = True
                document_copy["embedding_shape_0"] = int(embedding.shape[0])
                document_copy["embedding_shape_1"] = int(embedding.shape[1])
            else:
                document_copy = document.copy()
                document_copy["is_colbert"] = False
            self.client.table(collection_name).insert(document_copy).execute()
    def query(self, collection_name: str, query_vector, limit: int = 5, embedding_model=None):
        """
        Query the vector database. 
        query_vector can be either dense vector or ColBERT multi-vector
        embedding_model: needed for ColBERT similarity computation
        """
        is_query_colbert = isinstance(query_vector, np.ndarray) and len(query_vector.shape) == 2
        
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            
            if is_query_colbert:
                # For ColBERT queries, we need to retrieve all documents and compute similarity manually
                all_docs = list(collection.find({}))
                results = []
                
                for doc in all_docs:
                    if doc.get("is_colbert", False):
                        doc_embedding = pickle.loads(doc["embedding"])
                        score = embedding_model.compute_colbert_similarity(query_vector, doc_embedding)

                    else:
                        # Dense vector comparison (not ideal for ColBERT query)
                        query_mean = np.mean(query_vector, axis=0).tolist()
                        score = np.dot(query_mean, doc["embedding"])
                    
                    results.append({
                        "title": doc["title"],
                        "information": doc["information"],
                        "score": float(score)
                    })
                
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]
            else:
                # Original dense vector search
                results = collection.aggregate([
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "queryVector": query_vector,
                            "path": "embedding",
                            "numCandidates": 100,
                            "limit": limit
                        }
                    }
                ])
                return list(results)
                
        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name=collection_name)
            
            if is_query_colbert:
                # Get all documents for ColBERT similarity computation
                all_results = collection.get(include=['documents', 'metadatas', 'embeddings'])
                results = []
                
                for i in range(len(all_results['ids'])):
                    metadata = all_results['metadatas'][i]
                    if metadata.get("is_colbert", False):
                        doc_embedding = pickle.loads(bytes.fromhex(metadata["colbert_embedding"]))
                        score = embedding_model.compute_colbert_similarity(query_vector, doc_embedding)
                    else:
                        # Fallback to cosine similarity with mean pooling
                        query_mean = np.mean(query_vector, axis=0)
                        doc_vec = np.array(all_results['embeddings'][i])
                        score = np.dot(query_mean, doc_vec) / (np.linalg.norm(query_mean) * np.linalg.norm(doc_vec))
                    
                    results.append({
                        "title": all_results['ids'][i],
                        "information": all_results['documents'][i],
                        "score": float(score)
                    })
                
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]
            else:
                # Original dense vector search
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=limit
                )
                docs = []
                for i in range(len(results["ids"][0])):
                    docs.append({
                        "title": results["ids"][0][i],
                        "information": results["documents"][0][i]
                    })
                return docs
                
        elif self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Warning] Collection '{collection_name}' doesn't exist for querying")
                return []
            
            if is_query_colbert:
                # Retrieve all documents for ColBERT similarity computation
                all_results, _ = self.client.scroll(
                    collection_name=collection_name,
                    limit=10000  # Get all documents
                )
                
                results = []
                for point in all_results:
                    payload = point.payload
                    if payload.get("is_colbert", False):
                        doc_embedding = pickle.loads(bytes.fromhex(payload["colbert_embedding"]))
                        score = embedding_model.compute_colbert_similarity(query_vector, doc_embedding)
                    else:
                        # Fallback similarity
                        query_mean = np.mean(query_vector, axis=0)
                        score = np.dot(query_mean, point.vector)
                    
                    results.append({
                        "title": payload["title"],
                        "information": payload["information"],
                        "score": float(score)
                    })
                
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]
            else:
                # Original dense vector search
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result.payload["title"],
                        "information": result.payload["information"],
                        "score": result.score
                    })
                return formatted_results
                
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*").execute()
            
            if is_query_colbert:
                results = []
                for doc in response.data:
                    if doc.get("is_colbert", False):
                        doc_embedding = pickle.loads(bytes.fromhex(doc["embedding"]))
                        score = embedding_model.compute_colbert_similarity(query_vector, doc_embedding)
                    else:
                        query_mean = np.mean(query_vector, axis=0).tolist()
                        score = np.dot(query_mean, doc["embedding"])
                    
                    results.append({
                        "title": doc["title"],
                        "information": doc["information"],
                        "score": float(score)
                    })
                
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:limit]
            else:
                return response.data

    def document_exists(self, collection_name, filter_query):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            return collection.count_documents(filter_query) > 0
        elif self.db_type == "chromadb":
            try:
                collection = self.client.get_or_create_collection(name=collection_name)
                # Lấy toàn bộ ID hiện có trong collection
                all_ids = collection.get()["ids"]
                return filter_query["title"] in all_ids
            except Exception as e:
                print(f"Error checking existence in ChromaDB: {e}")
                return False
        elif self.db_type == "qdrant":
            if not self.client.collection_exists(collection_name=collection_name):
                print(f"[Info] Collection '{collection_name}' doesn't exist yet")
                return False
                
            # Search for document with matching title
            try:
                result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [
                            {
                                "key": "title",
                                "match": {"value": filter_query["title"]}
                            }
                        ]
                    },
                    limit=1
                )
                return len(result[0]) > 0
            except Exception as e:
                print(f"Error checking document existence in Qdrant: {e}")
                return False
        elif self.db_type == "supabase":
            response = self.client.table(collection_name).select("*").eq("title", filter_query["title"]).execute()
            return len(response.data) > 0
        else:
            raise ValueError("Unsupported database type")
    def count_documents(self, collection_name: str) -> int:
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            return collection.count_documents({})

        elif self.db_type == "chromadb":
            collection = self.client.get_or_create_collection(name=collection_name)
            return collection.count()

        elif self.db_type == "qdrant":
            result = self.client.count(collection_name=collection_name, exact=True)
            return result.count

        elif self.db_type == "supabase":
            data = self.client.table(collection_name).select("id", count="exact").execute()
            return data.count

        else:
            raise NotImplementedError("Vector DB type chưa hỗ trợ: " + self.db_type)

    def drop_collection(self, collection_name: str):
        if self.db_type == "mongodb":
            db = self.client.get_database("vector_db")
            collection = db[collection_name]
            collection.drop()
        elif self.db_type == "chromadb":
            self.client.delete_collection(name=collection_name)
        elif self.db_type == "qdrant":
            if self.client.collection_exists(collection_name=collection_name):
                self.client.delete_collection(collection_name=collection_name)
        elif self.db_type == "supabase":
            self.client.table(collection_name).delete().execute()
        else:
            raise ValueError("Unsupported database type for drop_collection")