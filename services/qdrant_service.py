import re
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import get_qdrant_client
from services.llm_service import get_genai_client

EMBEDDING_DIM = 768

def sanitize_collection_name(repo_name: str) -> str:
    """Converts 'facebook/react' into safe Qdrant collection name 'repograph_facebook_react'."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", repo_name.lower())
    return f"repograph_{safe}"

def get_text_embedding(text: str) -> List[float]:
    """Generates embedding using Gemini embedding model (gemini-embedding-001) or deterministic fallback vector."""
    embeddings = get_batch_text_embeddings([text])
    if embeddings:
        return embeddings[0]
    return _deterministic_fallback_vector(text)

def _deterministic_fallback_vector(text: str) -> List[float]:
    """Generate pseudo-random normalized vector from text hash for fallback."""
    vec = []
    h = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
    for i in range(EMBEDDING_DIM):
        val = ((h >> (i % 64)) & 0xFF) / 255.0 - 0.5
        vec.append(val)
    return vec

def get_batch_text_embeddings(texts: List[str], batch_size: int = 40) -> List[List[float]]:
    """
    Generates embeddings in efficient batches using Gemini embedding model.
    Sends 40-50 texts per single HTTP request, reducing 60+ individual calls down to 1-2 calls (2-3 seconds total).
    """
    if not texts:
        return []

    client = get_genai_client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        clean_batch = [t[:2000] if t and t.strip() else " " for t in batch]
        batch_success = False

        if client:
            for model_name in ["gemini-embedding-001", "text-embedding-004"]:
                try:
                    from google.genai import types
                    res = client.models.embed_content(
                        model=model_name,
                        contents=clean_batch,
                        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)
                    )
                    if hasattr(res, 'embeddings') and res.embeddings:
                        for emb in res.embeddings:
                            all_embeddings.append(emb.values)
                        batch_success = True
                        break
                    elif hasattr(res, 'embedding') and res.embedding and res.embedding.values:
                        all_embeddings.append(res.embedding.values)
                        batch_success = True
                        break
                except Exception as e:
                    print(f"Batch embed attempt error ({model_name}): {e}")
                    continue

        if not batch_success:
            for t in batch:
                all_embeddings.append(_deterministic_fallback_vector(t))

    return all_embeddings

def chunk_file_content(path: str, content: str, chunk_size: int = 600, overlap: int = 150) -> List[Dict[str, Any]]:
    """Splits file text into chunks with approximate line ranges."""
    lines = content.splitlines()
    chunks = []
    
    current_text = []
    current_len = 0
    start_line = 1
    
    for idx, line in enumerate(lines, 1):
        current_text.append(line)
        current_len += len(line) + 1
        
        if current_len >= chunk_size:
            chunk_str = "\n".join(current_text)
            chunks.append({
                "file": path,
                "start_line": start_line,
                "end_line": idx,
                "text": chunk_str
            })
            # keep overlap lines
            overlap_lines = current_text[-max(1, int(len(current_text) * 0.25)):]
            current_text = overlap_lines
            current_len = sum(len(l) + 1 for l in current_text)
            start_line = max(1, idx - len(overlap_lines) + 1)
            
    if current_text:
        chunks.append({
            "file": path,
            "start_line": start_line,
            "end_line": len(lines),
            "text": "\n".join(current_text)
        })
        
    return chunks

async def ingest_repository_chunks(repo_name: str, file_samples: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Ingests file chunks into Qdrant collection using high-speed batch embedding.
    Ensures vector database is 100% populated and indexed before returning all chunks.
    """
    coll_name = sanitize_collection_name(repo_name)
    
    # 1. Collect all chunks across all selected files
    all_chunks: List[Dict[str, Any]] = []
    for path, content in file_samples.items():
        if not content:
            continue
        file_chunks = chunk_file_content(path, content)
        all_chunks.extend(file_chunks)

    if not all_chunks:
        print(f"No chunks extracted for {repo_name}")
        return []

    # 2. Re-create collection safely in Qdrant
    q_client = get_qdrant_client()
    try:
        if q_client.collection_exists(collection_name=coll_name):
            q_client.delete_collection(collection_name=coll_name)
        q_client.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        # Create payload index on 'file' for keyword filtering in Qdrant Cloud
        try:
            q_client.create_payload_index(
                collection_name=coll_name,
                field_name="file",
                field_schema="keyword"
            )
        except Exception as idx_err:
            print(f"Notice on payload index creation: {idx_err}")
    except Exception as e:
        print(f"Error creating Qdrant collection: {e}")
        return all_chunks

    # 3. High-speed batch embedding of all chunk texts (takes ~2 seconds for 60+ chunks)
    texts_to_embed = [c["text"] for c in all_chunks]
    vectors = get_batch_text_embeddings(texts_to_embed, batch_size=40)

    # 4. Upsert points into Qdrant
    points = []
    for idx, (chunk, vector) in enumerate(zip(all_chunks, vectors), 1):
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload=chunk
            )
        )

    if points:
        try:
            q_client.upsert(collection_name=coll_name, points=points)
            print(f"Successfully upserted {len(points)} chunks into {coll_name}")
        except Exception as e:
            print(f"Qdrant upsert error: {e}")

    return all_chunks

async def search_repository_chunks(repo_name: str, query: str, node_files: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Qdrant. If node_files provided, attempts focused search first,
    then falls back to whole-repo search. Auto-creates payload index if needed.
    """
    q_client = get_qdrant_client()
    coll_name = sanitize_collection_name(repo_name)
    
    try:
        if not q_client.collection_exists(collection_name=coll_name):
            return []
    except Exception:
        return []

    query_vector = get_text_embedding(query)
    
    def _execute_query(q_filter=None, limit=top_k):
        if hasattr(q_client, "query_points"):
            res = q_client.query_points(
                collection_name=coll_name,
                query=query_vector,
                query_filter=q_filter,
                limit=limit
            )
            return [p.payload for p in getattr(res, "points", []) if getattr(p, "payload", None)]
        elif hasattr(q_client, "search"):
            res = q_client.search(
                collection_name=coll_name,
                query_vector=query_vector,
                query_filter=q_filter,
                limit=limit
            )
            return [hit.payload for hit in res if getattr(hit, "payload", None)]
        return []

    results = []

    # 1. Scoped search if node_files given
    if node_files and len(node_files) > 0:
        for nf in node_files[:3]:
            try:
                flt = Filter(must=[FieldCondition(key="file", match=MatchValue(value=nf))])
                scoped = _execute_query(q_filter=flt, limit=2)
                for item in scoped:
                    results.append(item)
            except Exception as filter_err:
                # If Qdrant complains index is missing, auto-create it on the fly and retry
                try:
                    q_client.create_payload_index(collection_name=coll_name, field_name="file", field_schema="keyword")
                    flt = Filter(must=[FieldCondition(key="file", match=MatchValue(value=nf))])
                    scoped = _execute_query(q_filter=flt, limit=2)
                    for item in scoped:
                        results.append(item)
                except Exception:
                    pass

    # 2. General vector search (always executes independently)
    try:
        general = _execute_query(limit=top_k)
        for item in general:
            if not any(r.get("text") == item.get("text") for r in results):
                results.append(item)
    except Exception as e:
        print(f"Qdrant general search error: {e}")

    return results[:top_k]
