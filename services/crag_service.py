"""
CRAG (Corrective RAG) Pipeline — RepoGraph AI
Strategy: REPO-FIRST. Use repository chunks whenever available.
Only fall back to web for explicitly external queries with zero repo context.
"""
import json
import warnings
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from config import TAVILY_API_KEY
from google.genai import types
from services.llm_service import get_genai_client, generate_trace_path
from services.qdrant_service import search_repository_chunks

# ── Single model: gemini-2.5-flash only ──────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── Explicit external-knowledge patterns (ONLY go to web for these) ───────────
EXTERNAL_PATTERNS = [
    "what is the latest version",
    "latest version of",
    "latest release of",
    "current stable version",
    "how to install ",
    "how do i install",
    "getting started with ",
    "official documentation",
    "what is kubernetes",
    "what is docker",
    "what is react",
    "what is python",
    "what is flutter",
    "what is django",
    "what is fastapi",
    "who created",
    "who made ",
    "when was it created",
    "history of ",
]


def _is_external_query(message: str) -> bool:
    """Returns True ONLY if the query is clearly about external world knowledge."""
    lower = message.lower().strip()
    return any(pat in lower for pat in EXTERNAL_PATTERNS)


def free_web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Performs web search using DuckDuckGo, falls back to Tavily."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                return [
                    {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                    for r in results
                ]
    except Exception as e:
        print(f"[CRAG] DuckDuckGo error: {e}")

    if TAVILY_API_KEY:
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    "https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results}
                )
                if res.status_code == 200:
                    return [
                        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                        for r in res.json().get("results", [])
                    ]
        except Exception as e:
            print(f"[CRAG] Tavily error: {e}")

    return []


async def run_crag_pipeline(
    repo_name: str,
    message: str,
    node_context: Optional[Dict[str, Any]] = None,
    all_nodes: Optional[List[Dict[str, Any]]] = None,
    all_edges: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    CRAG Pipeline — Repo-First Strategy:
    
    Decision tree (NO LLM evaluator — pure rule-based routing):
      1. Explicitly external query (version, install, what-is-X)?
         → YES: web search
         → NO:  continue
      2. Do we have Qdrant repo chunks?
         → YES: answer from repo (ALWAYS — no second-guessing)
         → NO:  do we have architecture nodes?
                → YES: answer from architecture context
                → NO:  web search as last resort
    
    The LLM evaluator was the bug — it kept returning insufficient=false.
    Removed entirely. Routing is now deterministic.
    """
    client = get_genai_client()
    node_files = node_context.get("files", []) if node_context else []
    lower_msg = message.lower().strip()

    # ── Step 1: Hard external-query check ────────────────────────────────────
    if _is_external_query(message):
        print(f"[CRAG] External query detected: '{message[:60]}' → web search")
        web_results = free_web_search(message, max_results=3)
        sources = [w["url"] for w in web_results if w.get("url")]
        answer = await _generate_answer(
            client=client,
            repo_name=repo_name,
            message=message,
            source_type="web",
            web_results=web_results,
            repo_context="",
            arch_block="",
            node_block="",
        )
        return {"answer": answer, "sources": sources, "source_type": "web", "highlighted_path": []}

    # ── Step 2: Retrieve repo chunks from Qdrant ─────────────────────────────
    repo_chunks = await search_repository_chunks(
        repo_name, message, node_files=node_files, top_k=10
    )
    print(f"[CRAG] Retrieved {len(repo_chunks)} chunks from Qdrant for repo '{repo_name}'")

    repo_context = "\n\n".join([
        f"### File: {c.get('file')} (Lines {c.get('start_line')}-{c.get('end_line')})\n{c.get('text', '')}"
        for c in repo_chunks
    ])

    # ── Step 3: Build architecture block ─────────────────────────────────────
    arch_block = ""
    is_arch_query = any(k in lower_msg for k in [
        "architecture", "system flow", "how does the app work", "how does the project work",
        "overview", "components", "modules", "call hierarchy", "structure", "explain repo",
        "how does this work", "what does this repo do", "what does this project do",
        "how is this built", "tech stack",
    ])
    if all_nodes:
        nodes_list = [
            f"- [{n.get('type', '').upper()}] {n.get('label')}: {n.get('description', '')}"
            for n in all_nodes
        ]
        edges_list = [
            f"- {e.get('source')} → {e.get('target')}: {e.get('label', 'calls')}"
            for e in (all_edges or [])
        ]
        # Always include architecture if nodes exist — it's always useful context
        arch_block = (
            f"\n## Repository Architecture\n"
            f"{chr(10).join(nodes_list)}\n"
            f"\n## Data Flow\n"
            f"{chr(10).join(edges_list)}\n"
        )

    # ── Step 4: Build node block ──────────────────────────────────────────────
    node_block = ""
    if node_context:
        node_block = (
            f"\n## Selected Component: {node_context.get('label')} ({node_context.get('type')})\n"
            f"Description: {node_context.get('description', '')}\n"
            f"Files: {', '.join(node_files)}\n"
        )

    # ── Step 5: REPO-FIRST routing decision ──────────────────────────────────
    if repo_chunks:
        # We have actual repo code — ALWAYS use it, no LLM evaluation needed
        print(f"[CRAG] Using repo context ({len(repo_chunks)} chunks) for answer")
        sources = [
            f"{c.get('file')}:{c.get('start_line')}-{c.get('end_line')}"
            for c in repo_chunks[:5]
        ]
        answer = await _generate_answer(
            client=client,
            repo_name=repo_name,
            message=message,
            source_type="repo",
            web_results=[],
            repo_context=repo_context,
            arch_block=arch_block,
            node_block=node_block,
        )
        source_type = "repo"

    elif all_nodes and len(all_nodes) > 0:
        # No Qdrant chunks but we have architecture graph — use it
        print(f"[CRAG] No Qdrant chunks; using architecture graph ({len(all_nodes)} nodes)")
        sources = [n.get("label", "") for n in all_nodes[:4]]
        answer = await _generate_answer(
            client=client,
            repo_name=repo_name,
            message=message,
            source_type="repo",
            web_results=[],
            repo_context="",
            arch_block=arch_block,
            node_block=node_block,
        )
        source_type = "repo"

    else:
        # Absolute last resort: web search
        print(f"[CRAG] No repo context at all — falling back to web")
        web_results = free_web_search(message, max_results=3)
        sources = [w["url"] for w in web_results if w.get("url")]
        answer = await _generate_answer(
            client=client,
            repo_name=repo_name,
            message=message,
            source_type="web",
            web_results=web_results,
            repo_context="",
            arch_block="",
            node_block=node_block,
        )
        source_type = "web"

    # ── Step 6: Trace path for flow/trace queries ─────────────────────────────
    highlighted_path = []
    if all_nodes and all_edges:
        if any(k in lower_msg for k in [
            "trace", "flow", "how does", "pipeline", "lifecycle", "login", "auth",
            "call chain", "execution", "request", "response",
        ]):
            node_id = (
                node_context.get("id", all_nodes[0]["id"]) if node_context
                else all_nodes[0]["id"]
            )
            highlighted_path = await generate_trace_path(node_id, message, all_nodes, all_edges)

    return {
        "answer": answer,
        "sources": sources,
        "source_type": source_type,
        "highlighted_path": highlighted_path,
    }


async def _generate_answer(
    client,
    repo_name: str,
    message: str,
    source_type: str,
    web_results: List[Dict],
    repo_context: str,
    arch_block: str,
    node_block: str,
) -> str:
    """Generates the final answer using gemini-2.5-flash."""
    if not client:
        if source_type == "web" and web_results:
            return "\n\n".join([
                f"**{r['title']}**\n{r['snippet']}" for r in web_results[:2]
            ])
        return f"Repository context loaded for **{repo_name}**. Ask me anything about the code."

    # Build context block
    if source_type == "web" and web_results:
        context_section = (
            "## Web Search Results\n"
            + "\n\n".join([f"**Source**: {r['url']}\n{r['snippet']}" for r in web_results])
        )
        instruction = (
            "Answer using ONLY the web search results above. "
            "Clearly state this is based on external documentation."
        )
    else:
        context_parts = []
        if node_block:
            context_parts.append(node_block)
        if arch_block:
            context_parts.append(arch_block)
        if repo_context:
            context_parts.append(f"## Repository Code Snippets\n{repo_context}")

        context_section = "\n".join(context_parts) if context_parts else "(No code context retrieved)"
        instruction = (
            "Answer DIRECTLY from the repository code snippets and architecture above. "
            "Reference specific file names, functions, classes, and line numbers you can see. "
            "Do NOT answer from general knowledge — only from the provided context."
        )

    prompt = f"""You are RepoGraph AI — an expert codebase analyst for the repository: **{repo_name}**

{context_section}

---
User Question: "{message}"

INSTRUCTIONS:
- {instruction}
- Use markdown: headers, bullet lists, inline `code`, and code blocks.
- Be concise and technical. Developers want direct, specific answers.
- If the provided context does not contain enough info to answer, say exactly what you found and what's missing."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )
        return response.text or "No response generated."
    except Exception as e:
        print(f"[CRAG] Generation error ({GEMINI_MODEL}): {e}")
        return f"Error generating answer: {e}. Please check your Gemini API key."
