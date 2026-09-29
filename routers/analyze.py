import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from config import GITHUB_TOKEN
from services.github_service import (
    parse_repo_url,
    get_repo_details,
    fetch_repo_tree,
    select_important_files,
    fetch_files_batch
)
from services.llm_service import generate_architecture_graph
from services.qdrant_service import ingest_repository_chunks

router = APIRouter()

class AnalyzeRequest(BaseModel):
    repo_url: str

class NodeItem(BaseModel):
    id: str
    label: str
    type: str  # 'frontend' | 'backend' | 'service' | 'database' | 'external'
    layer: Optional[int] = 0
    files: List[str] = []
    description: str = ""
    dependencies: List[str] = []
    used_by: List[str] = []

class EdgeItem(BaseModel):
    id: str
    source: str
    target: str
    label: Optional[str] = ""

class AnalyzeResponse(BaseModel):
    repo_name: str
    branch: str = "main"
    description: Optional[str] = ""
    stars: Optional[int] = 0
    language: Optional[str] = "Unknown"
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(req: AnalyzeRequest):
    try:
        owner, repo = parse_repo_url(req.repo_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    repo_full_name = f"{owner}/{repo}"
    
    # 1. Fetch metadata
    try:
        details = await get_repo_details(owner, repo, token=GITHUB_TOKEN)
        branch = details.get("default_branch", "main")
    except Exception as e:
        # Graceful fallback for rate limits or network issues
        details = {"description": "", "stars": 0, "language": "Unknown"}
        branch = "main"

    # 2. Fetch filtered file tree
    tree = await fetch_repo_tree(owner, repo, branch, token=GITHUB_TOKEN)
    file_paths = [item.get("path") for item in tree if item.get("type") == "blob"]
    
    # 3. Select important architectural files (16 key files)
    selected_files = select_important_files(tree, max_files=16)
    
    # 4. Fetch sample contents concurrently reusing connection pool
    file_samples = await fetch_files_batch(owner, repo, branch, selected_files, token=GITHUB_TOKEN)

    # 5. Ingest & Embed Chunks FIRST (High-Speed Batch Embedding into Qdrant Cloud)
    # Vectors are 100% indexed and ready before any graph or chatbot screen is displayed
    chunks = await ingest_repository_chunks(repo_full_name, file_samples)

    # 6. Synthesize Architecture Graph with Gemini grounded in the ingested chunks
    graph_data = await generate_architecture_graph(repo_full_name, file_paths, file_samples, chunks=chunks)

    return AnalyzeResponse(
        repo_name=repo_full_name,
        branch=branch,
        description=details.get("description") or "",
        stars=details.get("stars") or 0,
        language=details.get("language") or "Unknown",
        nodes=graph_data.get("nodes", []),
        edges=graph_data.get("edges", [])
    )
