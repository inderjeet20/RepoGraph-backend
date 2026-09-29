from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from services.crag_service import run_crag_pipeline

router = APIRouter()

class ChatRequest(BaseModel):
    repo_name: str
    message: str
    node_context: Optional[Dict[str, Any]] = None
    all_nodes: Optional[List[Dict[str, Any]]] = None
    all_edges: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    source_type: str  # 'repo' (blue) | 'web' (green)
    highlighted_path: List[str] = []

@router.post("/chat", response_model=ChatResponse)
async def chat_with_repo(req: ChatRequest):
    result = await run_crag_pipeline(
        repo_name=req.repo_name,
        message=req.message,
        node_context=req.node_context,
        all_nodes=req.all_nodes,
        all_edges=req.all_edges
    )
    
    return ChatResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        source_type=result.get("source_type", "repo"),
        highlighted_path=result.get("highlighted_path", [])
    )
