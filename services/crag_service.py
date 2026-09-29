import json
import warnings
from typing import List, Dict, Any, Optional

# Suppress ddgs package rename runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from config import TAVILY_API_KEY
from google.genai import types
from services.llm_service import get_genai_client, generate_trace_path
from services.qdrant_service import search_repository_chunks

def free_web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Performs completely free web search using DuckDuckGo.
    Falls back to Tavily if TAVILY_API_KEY is configured.
    """
    # 1. Try DuckDuckGo (100% free, no API key needed)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    for r in results
                ]
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")

    # 2. Fallback to Tavily if key present
    if TAVILY_API_KEY:
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results}
                )
                if res.status_code == 200:
                    data = res.json()
                    return [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("content", "")
                        }
                        for r in data.get("results", [])
                    ]
        except Exception as e:
            print(f"Tavily search error: {e}")

    return []

async def run_crag_pipeline(
    repo_name: str,
    message: str,
    node_context: Optional[Dict[str, Any]] = None,
    all_nodes: Optional[List[Dict[str, Any]]] = None,
    all_edges: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Corrective RAG (CRAG) execution:
    1. Retrieve repository chunks (focused by node_context if provided).
    2. Check relevance / sufficiency: detects if query requires external world/web knowledge or repo.
    3. If insufficient evidence in repo -> Fallback to Web Search via ddgs / Tavily.
    4. Generate structured, direct answer (without dumping unsolicited architecture).
    5. Optionally computes highlighted_path for focus mode.
    """
    client = get_genai_client()
    node_files = node_context.get("files", []) if node_context else []
    
    # Step 1: Retrieve repo chunks
    repo_chunks = await search_repository_chunks(repo_name, message, node_files=node_files, top_k=4)
    
    repo_context_text = "\n\n".join([
        f"File: {c.get('file')} (Lines {c.get('start_line')}-{c.get('end_line')}):\n{c.get('text')}"
        for c in repo_chunks
    ])

    lower_msg = message.lower().strip()
    is_arch_query = any(k in lower_msg for k in [
        "architecture", "system flow", "how does the app work", "how does the project work", 
        "overview", "components", "modules", "call hierarchy", "structure", "explain repo"
    ])

    # Step 2: Corrective RAG (CRAG) Relevance & Sufficiency Evaluation
    is_sufficient = False

    # If the user explicitly asks an architectural overview, local graph is sufficient
    if is_arch_query and all_nodes and len(all_nodes) > 0:
        is_sufficient = True
    elif client and repo_chunks:
        eval_prompt = f"""
You are the Retrieval Evaluator in a Corrective RAG (CRAG) system for codebase '{repo_name}'.
User Question: "{message}"

Retrieved Repository Code Snippets:
{repo_context_text[:1800]}

Evaluate whether the repository snippets above directly, accurately, and completely answer the question.
CRITICAL RULES:
1. If the user asks for external information (such as latest versions/releases of frameworks/languages, external documentation, third-party libraries, or general concepts) that is NOT explicitly stated in the code snippets:
   Answer {{"sufficient": false}}
2. If the snippets only show a local dependency (e.g. pubspec.yaml or package.json) but do NOT state what the latest external release is:
   Answer {{"sufficient": false}}
3. If the snippets do not directly and completely answer the question:
   Answer {{"sufficient": false}}
4. If the snippets directly, factually, and completely answer the question:
   Answer {{"sufficient": true}}

Output ONLY a JSON object: {{"sufficient": true}} or {{"sufficient": false}}
"""
        models_to_try = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
        for model in models_to_try:
            try:
                eval_res = client.models.generate_content(
                    model=model,
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                eval_data = json.loads(eval_res.text)
                is_sufficient = eval_data.get("sufficient", False)
                break
            except Exception as e:
                print(f"CRAG evaluation error ({model}): {e}")
                is_sufficient = False
    else:
        # No chunks retrieved or no client -> not sufficient
        is_sufficient = False

    # Step 3: Web Search Fallback if repo is not sufficient
    web_results = []
    source_type = "repo"
    sources = []

    if not is_sufficient:
        source_type = "web"
        # Search DuckDuckGo directly with the clean message
        web_results = free_web_search(message, max_results=3)
        sources = [w["url"] for w in web_results if w.get("url")]
        
        # If web search returned nothing, fall back to repo chunks if available
        if not sources and repo_chunks:
            source_type = "repo"
            sources = [f"{c.get('file')}:{c.get('start_line')}-{c.get('end_line')}" for c in repo_chunks[:2]]
    else:
        source_type = "repo"
        if repo_chunks:
            sources = [f"{c.get('file')}:{c.get('start_line')}-{c.get('end_line')}" for c in repo_chunks[:3]]
        elif node_files:
            sources = [f"{f}" for f in node_files]
        elif all_nodes:
            sources = [n.get("label", "Architecture") for n in all_nodes[:3]]

    # Step 4: Generation - Direct, concise, and focused
    answer = ""
    if client:
        context_block = ""
        if source_type == "web" and web_results:
            web_block = "\n\n".join([f"Source ({w['url']}):\n{w['snippet']}" for w in web_results])
            context_block += f"Web Search Results:\n{web_block}\n"
        elif repo_context_text and is_sufficient:
            context_block += f"Repository Code Snippets:\n{repo_context_text}\n"

        # Build architecture overview ONLY if the user specifically asked for flow/architecture
        arch_block = ""
        if is_arch_query and all_nodes:
            nodes_list = [f"- [{n.get('type', 'module').upper()}] {n.get('label')}: {n.get('description', '')}" for n in all_nodes]
            edges_list = [f"- {e.get('source')} -> {e.get('target')} ({e.get('label', 'calls')})" for e in (all_edges or [])]
            arch_block = f"\nArchitecture Layers:\n{chr(10).join(nodes_list)}\nFlow:\n{chr(10).join(edges_list)}\n"

        node_block = ""
        if node_context:
            node_block = f"Currently Selected Component: {node_context.get('label')} ({node_context.get('type')})\nDescription: {node_context.get('description')}\nFiles: {', '.join(node_files)}\n"

        prompt = f"""
You are RepoGraph AI, an expert software architecture and codebase copilot.
Repository: {repo_name}
{node_block}
{arch_block}
{context_block}

User Question: "{message}"

CRITICAL INSTRUCTIONS:
1. ANSWER DIRECTLY: Answer ONLY the user's specific question directly. Provide the core answer in the first sentence.
2. DO NOT DUMP UNRELATED ARCHITECTURE: Do NOT list unrelated components, controllers, database schemas, or routing packages unless explicitly asked for the full system architecture or flow.
3. CONCISE & PRECISE: Keep the explanation short, clean, and technical.
4. EXTERNAL VS REPO CLARITY: If the question is about external tool versions (e.g. latest Flutter version), state the latest external version from web search, and (if relevant) briefly mention what the local repo is configured with in 1 short sentence.
"""
        models_to_try = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-2.5-flash"]
        for model in models_to_try:
            try:
                gen_res = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                answer = gen_res.text
                if answer:
                    break
            except Exception as e:
                print(f"CRAG answer generation error ({model}): {e}")
                answer = "Unable to generate answer at this time."
    else:
        # Fallback offline answer
        if source_type == "web":
            answer = f"According to online documentation, here are the latest details regarding '{message}'."
        elif node_context:
            answer = f"The **{node_context.get('label')}** module ({node_context.get('type')}) defines: {node_context.get('description', '')}."
        else:
            answer = f"Repository overview for **{repo_name}**."

    # Step 5: Check if a trace path should be highlighted on the graph
    highlighted_path = []
    lower_msg = message.lower()
    if all_nodes and all_edges:
        if any(k in lower_msg for k in ["trace", "flow", "how does", "pipeline", "lifecycle", "work", "login", "auth"]):
            node_id = node_context.get("id", all_nodes[0]["id"]) if node_context else all_nodes[0]["id"]
            highlighted_path = await generate_trace_path(node_id, message, all_nodes, all_edges)

    return {
        "answer": answer,
        "sources": sources,
        "source_type": source_type,  # 'repo' (blue) or 'web' (green)
        "highlighted_path": highlighted_path
    }
