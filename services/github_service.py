import re
import asyncio
import httpx
from typing import Tuple, List, Dict, Any

GITHUB_API_BASE = "https://api.github.com"
RAW_GITHUB_BASE = "https://raw.githubusercontent.com"

def parse_repo_url(url: str) -> Tuple[str, str]:
    """
    Extracts (owner, repo) from GitHub URL or 'owner/repo' string.
    Example: 'https://github.com/facebook/react' -> ('facebook', 'react')
    """
    clean = url.strip().rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]
    
    match = re.search(r"github\.com/([^/]+)/([^/]+)", clean)
    if match:
        return match.group(1), match.group(2)
    
    parts = clean.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    
    raise ValueError(f"Invalid GitHub repository URL: {url}")

async def get_repo_details(owner: str, repo: str, token: str = "") -> Dict[str, Any]:
    """Fetches repository metadata including default branch."""
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "RepoGraph"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            res = await client.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", headers=headers)
            if res.status_code == 200:
                data = res.json()
                return {
                    "default_branch": data.get("default_branch") or "main",
                    "description": data.get("description") or "",
                    "stars": data.get("stargazers_count") or 0,
                    "language": data.get("language") or "Unknown",
                }
        except Exception:
            pass
            
    return {"default_branch": "main", "description": "", "stars": 0, "language": "Unknown"}

async def fetch_repo_tree(owner: str, repo: str, branch: str, token: str = "") -> List[Dict[str, Any]]:
    """Fetches recursive file tree from GitHub Git Trees API and filters non-code assets."""
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "RepoGraph"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                raw_tree = data.get("tree", [])

                ignored_dirs = (
                    "node_modules/", "vendor/", "dist/", "build/", ".git/", 
                    "__pycache__/", "tests/", "test/", ".github/", ".vscode/",
                    ".idea/", "venv/", ".venv/", ".dart_tool/",
                    "android/app/src/main/res/", "ios/Runner.xcworkspace/", 
                    "ios/Pods/", "windows/runner/resources/", "macos/"
                )
                ignored_extensions = (
                    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
                    ".mp4", ".mov", ".pdf", ".zip", ".tar", ".gz",
                    ".ttf", ".woff", ".woff2", ".eot",
                    ".lock", ".class", ".jar", ".map"
                )

                filtered = []
                for item in raw_tree:
                    if item.get("type") != "blob":
                        continue
                    p = item.get("path", "")
                    p_lower = p.lower()
                    if any(d in p_lower for d in ignored_dirs):
                        continue
                    if any(p_lower.endswith(ext) for ext in ignored_extensions):
                        continue
                    filtered.append(item)

                return filtered
        except Exception:
            pass
    return []

def select_important_files(tree: List[Dict[str, Any]], max_files: int = 16) -> List[str]:
    """
    Selects up to 16 most architecturally significant files for deep and accurate analysis.
    """
    candidates = []
    
    high_priority_names = {
        # Documentation & Architecture
        "readme.md", "readme.txt", "readme",
        # Package manifests & configs
        "package.json", "pubspec.yaml", "requirements.txt", "pyproject.toml", 
        "cargo.toml", "go.mod", "pom.xml", "build.gradle", "gemfile", "dockerfile", "docker-compose.yml",
        # Primary Entrypoints
        "main.dart", "app.dart", "main.py", "app.py", "main.ts", "main.js", 
        "index.ts", "index.js", "app.tsx", "app.jsx", "main.go", "main.rs", 
        "server.ts", "server.js", "server.py", "application.java"
    }
    
    structural_keywords = [
        "service", "controller", "router", "route", "handler", "model", "schema", 
        "entity", "repository", "store", "middleware", "auth", "api", "provider", 
        "bloc", "cubit", "notifier", "screen", "view", "page", "component",
        "pipeline", "agent", "workflow", "engine", "client", "network"
    ]

    for item in tree:
        path = item.get("path", "")
        path_lower = path.lower()
        filename = path_lower.split("/")[-1]
        
        # Priority 0: Documentation, Manifests, & Primary Entrypoints
        if filename in high_priority_names:
            candidates.append((0, path))
            continue
            
        # Priority 1: Key architectural modules (services, screens, controllers)
        if any(k in path_lower for k in structural_keywords):
            candidates.append((1, path))
            continue

        # Priority 2: General source files in core code folders
        if any(folder in path_lower for folder in ["lib/", "src/", "app/", "backend/"]):
            candidates.append((2, path))
            continue

    # Sort by priority, then shallowest path depth
    candidates.sort(key=lambda x: (x[0], len(x[1].split("/")), x[1]))
    selected = [c[1] for c in candidates[:max_files]]

    # Ensure README.md is always included if present in tree
    readme_item = next((item.get("path") for item in tree if item.get("path", "").lower() in ["readme.md", "readme"]), None)
    if readme_item and readme_item not in selected:
        selected.insert(0, readme_item)

    return selected[:max_files]

async def fetch_file_content(owner: str, repo: str, branch: str, path: str, token: str = "") -> str:
    """Fetches raw text of a file using raw.githubusercontent.com."""
    headers = {"User-Agent": "RepoGraph"}
    if token:
        headers["Authorization"] = f"token {token}"
        
    url = f"{RAW_GITHUB_BASE}/{owner}/{repo}/{branch}/{path}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            res = await client.get(url, headers=headers)
            if res.status_code == 200:
                return res.text[:3000]
        except Exception:
            pass
    return ""

async def fetch_files_batch(owner: str, repo: str, branch: str, paths: List[str], token: str = "") -> Dict[str, str]:
    """Fetches multiple files concurrently reusing an HTTP connection pool for fast analysis."""
    if not paths:
        return {}
    headers = {"User-Agent": "RepoGraph"}
    if token:
        headers["Authorization"] = f"token {token}"

    limits = httpx.Limits(max_keepalive_connections=20, max_connections=20)
    async with httpx.AsyncClient(timeout=5.0, limits=limits) as client:
        async def fetch_one(path: str):
            url = f"{RAW_GITHUB_BASE}/{owner}/{repo}/{branch}/{path}"
            try:
                res = await client.get(url, headers=headers)
                if res.status_code == 200:
                    # Give README and manifests up to 4000 chars, source files up to 2500 chars
                    limit = 4000 if any(k in path.lower() for k in ["readme", "pubspec", "package.json"]) else 2500
                    return path, res.text[:limit]
            except Exception:
                pass
            return path, ""

        tasks = [fetch_one(p) for p in paths]
        results = await asyncio.gather(*tasks)
        return {p: content for p, content in results if content}

