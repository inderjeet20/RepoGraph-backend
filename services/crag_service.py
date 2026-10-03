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


# Correct Gemini model names (ordered from fastest to most capable)
GEMINI_MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash"]


async def run_crag_pipeline(
    repo_name: str,
    message: str,
    node_context: Optional[Dict[str, Any]] = None,
    all_nodes: Optional[List[Dict[str, Any]]] = None,
    all_edges: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Corrective RAG (CRAG) execution with repo-first policy:
    1. Retrieve repository chunks (focused by node_context if provided).
    2. Evaluate sufficiency — biased toward repo when chunks exist.
    3. Fallback to web only for genuinely external queries.
    4. Generate structured, direct answer grounded in repo context.
    5. Optionally computes highlighted_path for focus mode.
    """
    client = get_genai_client()
    node_files = node_context.get("files", []) if node_context else []

    lower_msg = message.lower().strip()

    # Detect explicit external-knowledge queries (not about this repo)
    is_external_query = any(k in lower_msg for k in [
        "latest version", "latest release", "what is the latest", "current version of",
        "how to install", "tutorial for", "documentation for", "what is react", "what is python",
        "what is flutter", "who made", "when was", "history of"
    ])

    is_arch_query = any(k in lower_msg for k in [
        "architecture", "system flow", "how does the app work", "how does the project work",
        "overview", "components", "modules", "call hierarchy", "structure", "explain repo",
        "how does this work", "what does this repo do", "what does this project do"
    ])

    # Step 1: Retrieve repo chunks (more chunks = better evidence)
    repo_chunks = await search_repository_chunks(repo_name, message, node_files=node_files, top_k=8)

    repo_context_text = "\n\n".join([
        f"File: {c.get('file')} (Lines {c.get('start_line')}-{c.get('end_line')}):\n{c.get('text')}"
        for c in repo_chunks
    ])

    # Step 2: CRAG Sufficiency Evaluation (repo-first policy)
    is_sufficient = False

    if is_external_query:
        # Explicitly external question — go to web
        is_sufficient = False
    elif is_arch_query and all_nodes and len(all_nodes) > 0:
        # Architecture overview query with nodes — always sufficient from repo
        is_sufficient = True
    elif repo_chunks and len(repo_chunks) >= 2:
        # We have meaningful repo chunks — evaluate with LLM but default to True
        if client:
            eval_prompt = f"""You are a Retrieval Evaluator in a Corrective RAG system for codebase '{repo_name}'.
User Question: "{message}"

Retrieved Repository Code Snippets ({len(repo_chunks)} chunks):
{repo_context_text[:2500]}

Evaluate: Can these repository snippets help answer the user's question about THIS codebase?

RULES:
1. If snippets show relevant code/files/logic that pertains to this question → answer {{"sufficient": true}}
2. If the question asks about EXTERNAL tools/versions/documentation not specific to this repo → answer {{"sufficient": false}}
3. If the question is about this specific codebase and we have related code → answer {{"sufficient": true}}
4. When in doubt, prefer {{"sufficient": true}} — let the repo answer first.

Output ONLY JSON: {{"sufficient": true}} or {{"sufficient": false}}"""

            for model in GEMINI_MODELS:
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
                    is_sufficient = eval_data.get("sufficient", True)
                    break
                except Exception as e:
                    print(f"CRAG evaluation error ({model}): {e}")
                    # If eval fails but we have chunks, assume sufficient (repo-first)
                    is_sufficient = True
        else:
            # No LLM client but we have chunks — use them
            is_sufficient = True
    elif repo_chunks and len(repo_chunks) == 1:
        # Single chunk — still try to use it
        is_sufficient = True
    else:
        # No chunks at all
        is_sufficient = False

    # Step 3: Web Search Fallback only if repo is insufficient
    web_results = []
    source_type = "repo"
    sources = []

    if not is_sufficient:
        source_type = "web"
        web_results = free_web_search(message, max_results=3)
        sources = [w["url"] for w in web_results if w.get("url")]

        # If web search returned nothing, fall back to repo chunks if available
        if not sources and repo_chunks:
            source_type = "repo"
            is_sufficient = True
            sources = [f"{c.get('file')}:{c.get('start_line')}-{c.get('end_line')}" for c in repo_chunks[:3]]
    else:
        source_type = "repo"
        if repo_chunks:
            sources = [f"{c.get('file')}:{c.get('start_line')}-{c.get('end_line')}" for c in repo_chunks[:4]]
        elif node_files:
            sources = list(node_files[:3])
        elif all_nodes:
            sources = [n.get("label", "Architecture") for n in all_nodes[:3]]

    # Step 4: Generation — grounded, direct, concise
    answer = ""
    if client:
        context_block = ""
        if source_type == "web" and web_results:
            web_block = "\n\n".join([f"Source ({w['url']}):\n{w['snippet']}" for w in web_results])
            context_block += f"Web Search Results:\n{web_block}\n"
        elif repo_context_text and is_sufficient:
            context_block += f"Repository Code Snippets:\n{repo_context_text}\n"

        arch_block = ""
        if is_arch_query and all_nodes:
            nodes_list = [
                f"- [{n.get('type', 'module').upper()}] {n.get('label')}: {n.get('description', '')}"
                for n in all_nodes
            ]
            edges_list = [
                f"- {e.get('source')} -> {e.get('target')} ({e.get('label', 'calls')})"
                for e in (all_edges or [])
            ]
            arch_block = f"\nArchitecture:\n{chr(10).join(nodes_list)}\nData Flow:\n{chr(10).join(edges_list)}\n"

        node_block = ""
        if node_context:
            node_block = (
                f"Selected Component: {node_context.get('label')} ({node_context.get('type')})\n"
                f"Description: {node_context.get('description')}\n"
                f"Files: {', '.join(node_files)}\n"
            )

        prompt = f"""You are RepoGraph AI, an expert software architecture analyst for the repository: {repo_name}
{node_block}
{arch_block}
{context_block}

User Question: "{message}"

INSTRUCTIONS:
1. Answer DIRECTLY and SPECIFICALLY from the repository code and architecture provided above.
2. Reference actual file names, functions, classes, and code patterns you can see in the context.
3. Use markdown formatting — headers, bullet lists, inline code (\`code\`), and code blocks when showing code.
4. Be concise but complete. Developers want direct answers, not padding.
5. If context is from web search, clearly say "Based on external documentation:".
6. Do NOT fabricate details not present in the provided context."""

        for model in GEMINI_MODELS:
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
                answer = "Unable to generate answer at this time. Please check your API key configuration."
    else:
        if source_type == "web":
            answer = f"According to online documentation, here are the latest details regarding '{message}'."
        elif node_context:
            answer = f"The **{node_context.get('label')}** module ({node_context.get('type')}) defines: {node_context.get('description', '')}."
        else:
            answer = f"Repository overview for **{repo_name}**."

    # Step 5: Trace path highlight for flow/trace queries
    highlighted_path = []
    if all_nodes and all_edges:
        if any(k in lower_msg for k in ["trace", "flow", "how does", "pipeline", "lifecycle", "work", "login", "auth"]):
            node_id = node_context.get("id", all_nodes[0]["id"]) if node_context else all_nodes[0]["id"]
            highlighted_path = await generate_trace_path(node_id, message, all_nodes, all_edges)

    return {
        "answer": answer,
        "sources": sources,
        "source_type": source_type,
        "highlighted_path": highlighted_path
    }
