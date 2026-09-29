import json
import re
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

def get_genai_client() -> Optional[genai.Client]:
    if GEMINI_API_KEY:
        try:
            return genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Error initializing GenAI Client: {e}")
    return None

async def generate_architecture_graph(
    repo_name: str,
    file_tree: List[str],
    file_samples: Dict[str, str],
    chunks: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Uses Gemini to analyze comprehensive repository context and chunks to synthesize an accurate,
    sequential architectural flow graph with explicit layers and directed dependencies.
    """
    client = get_genai_client()
    
    if client:
        # Separate documentation, manifests, and code snippets
        readme_content = ""
        manifest_content = ""
        code_snippets = {}

        for path, content in file_samples.items():
            p_lower = path.lower()
            if "readme" in p_lower:
                readme_content = content[:3500]
            elif any(m in p_lower for m in ["package.json", "pubspec.yaml", "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod"]):
                manifest_content = content[:2500]
            else:
                code_snippets[path] = content[:1800]

        # Extract high-signal chunk excerpts grounded in the indexed chunks
        chunk_excerpts = []
        if chunks:
            seen_files = set()
            for ch in chunks:
                f = ch.get("file", "")
                if f not in seen_files and len(chunk_excerpts) < 10:
                    seen_files.add(f)
                    snippet = ch.get("text", "")[:300].strip().replace("\n", " ")
                    chunk_excerpts.append(f"- {f} (lines {ch.get('start_line')}-{ch.get('end_line')}): {snippet}...")

        prompt = f"""
You are a Principal Software Architect. Perform a deep architectural analysis of the GitHub repository '{repo_name}'
and synthesize its true runtime and data flow into a clean, directed dependency graph based on its actual source code chunks.

=== 1. DOCUMENTATION & PURPOSE ===
{readme_content or "No README provided."}

=== 2. MANIFEST & TECH STACK DEPENDENCIES ===
{manifest_content or "No manifest provided."}

=== 3. CLEAN SOURCE FILE TREE ===
{json.dumps(file_tree[:200], indent=2)}

=== 4. KEY ARCHITECTURAL CODE SNIPPETS & CHUNKS ===
{json.dumps(code_snippets, indent=2)}

=== 5. EXTRACTED CODEBASE CHUNK EXCERPTS ===
{chr(10).join(chunk_excerpts) if chunk_excerpts else "Code extracts provided above."}

=== ARCHITECTURAL GRAPH REQUIREMENTS ===
1. FLOW ORDER & LAYERS:
   You MUST establish the true sequential execution flow of this application from top to bottom.
   Every node MUST contain an integer `layer` attribute indicating its depth in the call hierarchy:
   - Layer 0: Primary Entrypoint / User Interface / Client / CLI (e.g. Mobile UI, Web App, CLI Runner, Gateway)
   - Layer 1: Dispatcher / Routing / State Management / Controllers (e.g. BLoC/Providers, API Routers, Controllers)
   - Layer 2: Core Domain Logic / Business Services / Algorithms (e.g. AI Prompt Processing, Image Recognition, Auth Service, Calculation Engines)
   - Layer 3+: Persistence / Databases / Caches / External Integrations / Third-Party APIs (e.g. SQLite/Hive/Postgres, Gemini API, Cloud Storage)

2. DIRECTED EDGES (FLOW):
   - Edges MUST strictly follow the direction of invocation or data passing: `source` calls or provides data to `target`.
   - The `source` node must be upstream (layer(source) <= layer(target)).
   - Every edge MUST have a concise, meaningful `label` (e.g., "submits meal photo", "invokes Gemini Vision", "queries log history", "persists user token").

3. ACCURACY & SPECIFICITY GROUNDED IN CHUNKS:
   - Synthesize 4 to 8 nodes that accurately represent THIS specific project.
   - In each node's `files` list, include the actual file paths from the provided file tree and chunks.
   - Node Types must be one of: "frontend", "backend", "service", "database", "external".

Return ONLY valid JSON matching this exact structure:
{{
  "nodes": [
    {{
      "id": "lowercase_unique_id",
      "label": "Human Readable Component Name",
      "type": "frontend | backend | service | database | external",
      "layer": 0,
      "files": ["lib/screens/home_screen.dart", "lib/screens/camera_screen.dart"],
      "description": "Allows users to capture meal photos and view nutrition logs.",
      "dependencies": ["Food Recognition Service"],
      "used_by": []
    }}
  ],
  "edges": [
    {{
      "id": "e_source_target",
      "source": "source_node_id",
      "target": "target_node_id",
      "label": "action or data flow description"
    }}
  ]
}}
"""
        models_to_try = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-3.5-flash"]
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                data = json.loads(response.text)
                if "nodes" in data and len(data["nodes"]) > 0:
                    # Self-correction and architectural validation pass before returning
                    return validate_and_correct_graph(data, file_tree)
            except Exception as e:
                print(f"Gemini architecture generation error ({model}): {e}.")
        print("Falling back to heuristic architecture builder.")

    # Fallback heuristic architecture builder (ensures zero crash even without key)
    return validate_and_correct_graph(fallback_architecture_builder(repo_name, file_tree), file_tree)

def validate_and_correct_graph(data: Dict[str, Any], file_tree: List[str]) -> Dict[str, Any]:
    """
    Validates and self-corrects the architecture graph before showing it to the user:
    1. Normalizes node IDs and types.
    2. Enforces layer non-negativity and correct fallback layers.
    3. Eliminates phantom edges (where source or target doesn't exist).
    4. Guarantees Layer 0 entrypoint exists.
    """
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    if not raw_nodes:
        return fallback_architecture_builder("", file_tree)

    valid_types = {"frontend", "backend", "service", "database", "external"}
    type_layer_defaults = {"frontend": 0, "backend": 1, "service": 2, "database": 3, "external": 3}

    cleaned_nodes = []
    valid_ids = set()

    for idx, n in enumerate(raw_nodes):
        node_id = str(n.get("id") or f"node_{idx}").strip().lower().replace(" ", "_")
        valid_ids.add(node_id)
        
        n_type = str(n.get("type", "service")).lower()
        if n_type not in valid_types:
            n_type = "service"
            
        layer = n.get("layer")
        if not isinstance(layer, int) or layer < 0:
            layer = type_layer_defaults.get(n_type, 1)

        # Match files against actual file tree where possible
        files = n.get("files", [])
        if not isinstance(files, list):
            files = [str(files)]
        verified_files = [f for f in files if isinstance(f, str) and (f in file_tree or any(f in t for t in file_tree))]
        if not verified_files and files:
            verified_files = [files[0]]

        cleaned_nodes.append({
            "id": node_id,
            "label": str(n.get("label") or node_id.replace("_", " ").title()),
            "type": n_type,
            "layer": layer,
            "files": verified_files,
            "description": str(n.get("description") or f"Component managing {n_type} responsibilities."),
            "dependencies": n.get("dependencies", []),
            "used_by": n.get("used_by", [])
        })

    # Ensure at least one Layer 0 entrypoint
    has_layer_0 = any(n["layer"] == 0 for n in cleaned_nodes)
    if not has_layer_0 and cleaned_nodes:
        cleaned_nodes[0]["layer"] = 0
        cleaned_nodes[0]["type"] = "frontend"

    # Prune and correct edges
    cleaned_edges = []
    for idx, e in enumerate(raw_edges):
        src = str(e.get("source", "")).strip().lower().replace(" ", "_")
        tgt = str(e.get("target", "")).strip().lower().replace(" ", "_")
        if src in valid_ids and tgt in valid_ids and src != tgt:
            cleaned_edges.append({
                "id": str(e.get("id") or f"e_{src}_{tgt}_{idx}"),
                "source": src,
                "target": tgt,
                "label": str(e.get("label") or "interacts with")
            })

    return {"nodes": cleaned_nodes, "edges": cleaned_edges}

def fallback_architecture_builder(repo_name: str, file_tree: List[str]) -> Dict[str, Any]:
    """Generates structured architectural nodes based on file paths and keywords with proper layers."""
    nodes = []
    edges = []
    
    # Categorize files
    fe_files = [f for f in file_tree if any(k in f.lower() for k in ["ui", "component", "screen", "page", "view", "frontend", "app"])]
    api_files = [f for f in file_tree if any(k in f.lower() for k in ["api", "router", "route", "controller", "endpoint", "bloc", "provider"])]
    auth_files = [f for f in file_tree if any(k in f.lower() for k in ["auth", "jwt", "session", "user", "security"])]
    srv_files = [f for f in file_tree if any(k in f.lower() for k in ["service", "core", "logic", "util", "helper", "engine", "ai", "model"])]
    db_files = [f for f in file_tree if any(k in f.lower() for k in ["db", "database", "model", "schema", "store", "repository", "sql", "hive"])]

    # 1. Frontend / Client (Layer 0)
    nodes.append({
        "id": "client_ui",
        "label": "Client UI / Screens",
        "type": "frontend",
        "layer": 0,
        "files": fe_files[:4] or ["src/App.tsx", "lib/main.dart"],
        "description": "User interface, client screens, and user interaction entrypoints.",
        "dependencies": ["Routing & Controllers"],
        "used_by": []
    })

    # 2. Dispatcher / Gateway (Layer 1)
    nodes.append({
        "id": "api_gateway",
        "label": "Routing & Controllers",
        "type": "backend",
        "layer": 1,
        "files": api_files[:4] or ["src/api/routes.ts"],
        "description": "Routes incoming interactions and dispatches requests to domain logic.",
        "dependencies": ["Domain Services"],
        "used_by": ["Client UI / Screens"]
    })
    edges.append({"id": "e_ui_api", "source": "client_ui", "target": "api_gateway", "label": "dispatches user action"})

    # 3. Domain Services (Layer 2)
    nodes.append({
        "id": "core_services",
        "label": "Domain Logic & Services",
        "type": "service",
        "layer": 2,
        "files": srv_files[:4] or ["src/services/main.ts"],
        "description": "Executes core business logic, computational workflows, and data orchestration.",
        "dependencies": ["Data Store / Persistence"],
        "used_by": ["Routing & Controllers"]
    })
    edges.append({"id": "e_api_srv", "source": "api_gateway", "target": "core_services", "label": "invokes service logic"})

    # 4. Persistence Layer (Layer 3)
    nodes.append({
        "id": "database",
        "label": "Data Store & Persistence",
        "type": "database",
        "layer": 3,
        "files": db_files[:4] or ["src/models/schema.ts"],
        "description": "Handles local storage, schemas, database queries, and cached state.",
        "dependencies": [],
        "used_by": ["Domain Logic & Services"]
    })
    edges.append({"id": "e_srv_db", "source": "core_services", "target": "database", "label": "persists state & queries"})

    return {"nodes": nodes, "edges": edges}

async def generate_trace_path(node_id: str, flow_query: str, all_nodes: List[Dict[str, Any]], all_edges: List[Dict[str, Any]]) -> List[str]:
    """
    Returns an ordered list of node IDs that form the visual trace flow.
    E.g., for auth -> ['client_ui', 'api_gateway', 'auth_service', 'database']
    """
    node_ids = [n["id"] for n in all_nodes]
    client = get_genai_client()
    
    if client:
        prompt = f"""
Given this architecture:
Nodes: {json.dumps(all_nodes, indent=2)}
Edges: {json.dumps(all_edges, indent=2)}

The user clicked node '{node_id}' and requested a trace for: '{flow_query}'.
Return an ordered JSON array of node IDs that represent this execution sequence from start to finish.
Example: ["client_ui", "api_gateway", "auth_service", "database"]
Only include node IDs from the provided graph. Return ONLY the JSON array.
"""
        models_to_try = ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-3.5-flash"]
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                path = json.loads(response.text)
                if isinstance(path, list) and all(p in node_ids for p in path):
                    return path
            except Exception as e:
                print(f"Trace path error ({model}): {e}")

    # Fallback BFS/DFS path finder
    path = []
    if "client_ui" in node_ids:
        path.append("client_ui")
    if "api_gateway" in node_ids:
        path.append("api_gateway")
    if node_id not in path and node_id in node_ids:
        path.append(node_id)
    if "database" in node_ids and "database" not in path:
        path.append("database")
    return path
