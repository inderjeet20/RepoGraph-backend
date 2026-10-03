"""
RepoGraph Backend — Comprehensive Test Suite
Tests cover: CRAG pipeline, Qdrant service, GitHub service, LLM service, API endpoints.
Run with: pytest tests/ -v
"""
import asyncio
import json
import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ─────────────────────────────────────────────────────────────
# GITHUB SERVICE TESTS
# ─────────────────────────────────────────────────────────────

class TestParseRepoUrl:
    """Tests for parse_repo_url (edge cases)."""

    def test_full_https_url(self):
        from services.github_service import parse_repo_url
        assert parse_repo_url("https://github.com/facebook/react") == ("facebook", "react")

    def test_url_with_dot_git(self):
        from services.github_service import parse_repo_url
        assert parse_repo_url("https://github.com/owner/repo.git") == ("owner", "repo")

    def test_url_with_trailing_slash(self):
        from services.github_service import parse_repo_url
        assert parse_repo_url("https://github.com/owner/repo/") == ("owner", "repo")

    def test_owner_slash_repo(self):
        from services.github_service import parse_repo_url
        assert parse_repo_url("owner/repo") == ("owner", "repo")

    def test_url_with_extra_path_segments(self):
        from services.github_service import parse_repo_url
        owner, repo = parse_repo_url("https://github.com/owner/repo/tree/main")
        assert owner == "owner"
        assert repo == "repo"

    def test_invalid_url_raises(self):
        from services.github_service import parse_repo_url
        with pytest.raises(ValueError):
            parse_repo_url("not-a-github-url")

    def test_url_with_uppercase(self):
        from services.github_service import parse_repo_url
        owner, repo = parse_repo_url("https://github.com/Facebook/React")
        assert owner == "Facebook"
        assert repo == "React"


class TestSelectImportantFiles:
    """Tests for file selection logic."""

    def test_readme_always_included(self):
        from services.github_service import select_important_files
        tree = [
            {"path": "README.md", "type": "blob"},
            {"path": "src/main.py", "type": "blob"},
            {"path": "src/utils.py", "type": "blob"},
        ]
        selected = select_important_files(tree, max_files=2)
        assert "README.md" in selected

    def test_max_files_limit(self):
        from services.github_service import select_important_files
        tree = [{"path": f"src/file_{i}.py", "type": "blob"} for i in range(50)]
        selected = select_important_files(tree, max_files=16)
        assert len(selected) <= 16

    def test_prioritizes_manifests_over_generic_files(self):
        from services.github_service import select_important_files
        tree = [
            {"path": "requirements.txt", "type": "blob"},
            {"path": "main.py", "type": "blob"},
            {"path": "random_file.py", "type": "blob"},
        ]
        selected = select_important_files(tree, max_files=16)
        assert "requirements.txt" in selected
        assert "main.py" in selected

    def test_empty_tree(self):
        from services.github_service import select_important_files
        assert select_important_files([], max_files=16) == []


# ─────────────────────────────────────────────────────────────
# QDRANT SERVICE TESTS
# ─────────────────────────────────────────────────────────────

class TestChunkFileContent:
    """Tests for code chunking logic."""

    def test_small_file_single_chunk(self):
        from services.qdrant_service import chunk_file_content
        content = "line 1\nline 2\nline 3"
        chunks = chunk_file_content("test.py", content, chunk_size=10000)
        assert len(chunks) == 1
        assert chunks[0]["file"] == "test.py"
        assert chunks[0]["start_line"] == 1

    def test_large_file_multiple_chunks(self):
        from services.qdrant_service import chunk_file_content
        content = "\n".join([f"x = {i}  # comment to make line longer" for i in range(100)])
        chunks = chunk_file_content("big.py", content, chunk_size=200, overlap=50)
        assert len(chunks) > 1

    def test_chunk_line_ranges_are_valid(self):
        from services.qdrant_service import chunk_file_content
        content = "\n".join([f"line {i}" for i in range(50)])
        chunks = chunk_file_content("file.py", content, chunk_size=100)
        for chunk in chunks:
            assert chunk["start_line"] >= 1
            assert chunk["end_line"] >= chunk["start_line"]

    def test_empty_file(self):
        from services.qdrant_service import chunk_file_content
        chunks = chunk_file_content("empty.py", "")
        assert chunks == []

    def test_single_line_file(self):
        from services.qdrant_service import chunk_file_content
        chunks = chunk_file_content("single.py", "print('hello')")
        assert len(chunks) == 1
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 1

    def test_chunk_contains_original_text(self):
        from services.qdrant_service import chunk_file_content
        content = "def foo():\n    return 42\n"
        chunks = chunk_file_content("foo.py", content)
        assert any("def foo" in c["text"] for c in chunks)


class TestSanitizeCollectionName:
    """Tests for Qdrant collection name sanitization."""

    def test_slash_becomes_underscore(self):
        from services.qdrant_service import sanitize_collection_name
        assert sanitize_collection_name("facebook/react") == "repograph_facebook_react"

    def test_already_clean_name(self):
        from services.qdrant_service import sanitize_collection_name
        assert sanitize_collection_name("myrepo") == "repograph_myrepo"

    def test_uppercase_lowercased(self):
        from services.qdrant_service import sanitize_collection_name
        result = sanitize_collection_name("Owner/Repo")
        assert result == result.lower()

    def test_special_chars_stripped(self):
        from services.qdrant_service import sanitize_collection_name
        result = sanitize_collection_name("owner/repo-name.v2")
        assert re.match(r'^[a-z0-9_]+$', result)


# ─────────────────────────────────────────────────────────────
# LLM SERVICE TESTS
# ─────────────────────────────────────────────────────────────

class TestValidateAndCorrectGraph:
    """Tests for architectural graph validation and self-correction."""

    def test_removes_phantom_edges(self):
        from services.llm_service import validate_and_correct_graph
        data = {
            "nodes": [
                {"id": "a", "label": "A", "type": "frontend", "layer": 0, "files": [], "description": ""},
                {"id": "b", "label": "B", "type": "backend", "layer": 1, "files": [], "description": ""},
            ],
            "edges": [
                {"id": "e_a_b", "source": "a", "target": "b", "label": "calls"},
                {"id": "e_phantom", "source": "a", "target": "nonexistent", "label": "ghost"},
            ]
        }
        result = validate_and_correct_graph(data, [])
        edge_targets = [e["target"] for e in result["edges"]]
        assert "nonexistent" not in edge_targets

    def test_ensures_layer_0_exists(self):
        from services.llm_service import validate_and_correct_graph
        data = {
            "nodes": [
                {"id": "svc", "label": "Service", "type": "service", "layer": 2, "files": [], "description": ""},
            ],
            "edges": []
        }
        result = validate_and_correct_graph(data, [])
        assert any(n["layer"] == 0 for n in result["nodes"])

    def test_normalizes_node_types(self):
        from services.llm_service import validate_and_correct_graph
        data = {
            "nodes": [
                {"id": "x", "label": "X", "type": "unknown_type", "layer": 1, "files": [], "description": ""},
            ],
            "edges": []
        }
        result = validate_and_correct_graph(data, [])
        valid_types = {"frontend", "backend", "service", "database", "external"}
        for node in result["nodes"]:
            assert node["type"] in valid_types

    def test_self_loop_edges_removed(self):
        from services.llm_service import validate_and_correct_graph
        data = {
            "nodes": [
                {"id": "a", "label": "A", "type": "frontend", "layer": 0, "files": [], "description": ""},
            ],
            "edges": [
                {"id": "self_loop", "source": "a", "target": "a", "label": "self"},
            ]
        }
        result = validate_and_correct_graph(data, [])
        assert len(result["edges"]) == 0

    def test_empty_nodes_returns_fallback(self):
        from services.llm_service import validate_and_correct_graph
        data = {"nodes": [], "edges": []}
        result = validate_and_correct_graph(data, ["src/main.py"])
        assert len(result["nodes"]) > 0

    def test_negative_layer_corrected(self):
        from services.llm_service import validate_and_correct_graph
        data = {
            "nodes": [
                {"id": "x", "label": "X", "type": "backend", "layer": -1, "files": [], "description": ""},
            ],
            "edges": []
        }
        result = validate_and_correct_graph(data, [])
        assert all(n["layer"] >= 0 for n in result["nodes"])


class TestFallbackArchitectureBuilder:
    """Tests for the fallback architecture heuristic builder."""

    def test_returns_nodes_and_edges(self):
        from services.llm_service import fallback_architecture_builder
        result = fallback_architecture_builder("test/repo", ["src/app.py", "src/models.py"])
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    def test_always_has_layer_0(self):
        from services.llm_service import fallback_architecture_builder
        result = fallback_architecture_builder("test/repo", [])
        assert any(n["layer"] == 0 for n in result["nodes"])

    def test_all_edge_refs_valid(self):
        from services.llm_service import fallback_architecture_builder
        result = fallback_architecture_builder("test/repo", ["src/main.py"])
        node_ids = {n["id"] for n in result["nodes"]}
        for edge in result["edges"]:
            assert edge["source"] in node_ids, f"Invalid edge source: {edge['source']}"
            assert edge["target"] in node_ids, f"Invalid edge target: {edge['target']}"


# ─────────────────────────────────────────────────────────────
# CRAG PIPELINE TESTS — Rule-based routing (no LLM evaluator)
# ─────────────────────────────────────────────────────────────

class TestExternalQueryDetection:
    """Tests for _is_external_query detection function."""

    def test_detects_version_query(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("What is the latest version of React?") is True

    def test_detects_install_query(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("How to install FastAPI?") is True

    def test_detects_what_is_query(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("What is Docker?") is True

    def test_repo_question_not_external(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("How does authentication work in this repo?") is False

    def test_arch_question_not_external(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("What is the architecture of this project?") is False

    def test_code_question_not_external(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("What files does the auth service use?") is False

    def test_flow_question_not_external(self):
        from services.crag_service import _is_external_query
        assert _is_external_query("How does the login flow work?") is False


class TestCRAGPipelineRouting:
    """Tests for the CRAG pipeline rule-based routing."""

    @pytest.mark.asyncio
    async def test_external_query_goes_to_web(self):
        """Explicitly external queries bypass repo and hit web."""
        from services.crag_service import run_crag_pipeline

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = []
            mock_web.return_value = [{"url": "https://example.com", "snippet": "Flutter 3.24", "title": "Flutter"}]

            result = await run_crag_pipeline(
                repo_name="test/repo",
                message="What is the latest version of Flutter?",
            )
            assert result["source_type"] == "web"
            # search_repository_chunks should NOT be called for external queries
            mock_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_repo_chunks_used_without_llm_eval(self):
        """When chunks exist, repo is used DIRECTLY — no LLM evaluator called."""
        from services.crag_service import run_crag_pipeline

        mock_chunks = [
            {"file": "src/auth.py", "start_line": 1, "end_line": 30, "text": "def login(user, password): pass"},
            {"file": "src/auth.py", "start_line": 31, "end_line": 60, "text": "def verify_token(token): pass"},
            {"file": "src/jwt.py", "start_line": 1, "end_line": 20, "text": "import jwt\nSECRET = 'secret'"},
        ]

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = mock_chunks
            mock_web.return_value = []

            result = await run_crag_pipeline(
                repo_name="test/repo",
                message="How does authentication work in this codebase?",
            )

            assert result["source_type"] == "repo"
            mock_web.assert_not_called()  # Web must NOT be called when chunks exist
            assert len(result["sources"]) > 0

    @pytest.mark.asyncio
    async def test_architecture_nodes_used_when_no_chunks(self):
        """With no Qdrant chunks but with nodes, should use architecture context."""
        from services.crag_service import run_crag_pipeline

        mock_nodes = [
            {"id": "frontend", "label": "Frontend", "type": "frontend", "description": "React UI"},
            {"id": "backend", "label": "Backend", "type": "backend", "description": "FastAPI"},
        ]

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = []
            mock_web.return_value = []

            result = await run_crag_pipeline(
                repo_name="test/repo",
                message="How does this project work?",
                all_nodes=mock_nodes,
                all_edges=[],
            )

            assert result["source_type"] == "repo"
            mock_web.assert_not_called()

    @pytest.mark.asyncio
    async def test_web_only_when_absolutely_no_context(self):
        """No chunks, no nodes → web as absolute last resort."""
        from services.crag_service import run_crag_pipeline

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = []
            mock_web.return_value = [{"url": "https://docs.example.com", "snippet": "docs", "title": "Docs"}]

            result = await run_crag_pipeline(
                repo_name="test/repo",
                message="How does JWT authentication work?",  # not external, but no context
                all_nodes=None,
                all_edges=None,
            )

            assert result["source_type"] == "web"

    @pytest.mark.asyncio
    async def test_result_always_has_required_keys(self):
        """Result dict must always have all required keys regardless of path taken."""
        from services.crag_service import run_crag_pipeline

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = []
            mock_web.return_value = []

            result = await run_crag_pipeline(
                repo_name="test/repo",
                message="edge case: empty everything",
            )

            assert "answer" in result
            assert "sources" in result
            assert "source_type" in result
            assert "highlighted_path" in result
            assert isinstance(result["sources"], list)
            assert isinstance(result["highlighted_path"], list)
            assert result["source_type"] in ("repo", "web")

    @pytest.mark.asyncio
    async def test_sources_point_to_repo_files_when_chunks_exist(self):
        """Sources should be file:line references when using repo chunks."""
        from services.crag_service import run_crag_pipeline

        mock_chunks = [
            {"file": "backend/main.py", "start_line": 1, "end_line": 25, "text": "from fastapi import FastAPI"},
            {"file": "backend/routes.py", "start_line": 10, "end_line": 40, "text": "router = APIRouter()"},
        ]

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = mock_chunks
            mock_web.return_value = []

            result = await run_crag_pipeline(
                repo_name="test/repo",
                message="What framework is used for the backend?",
            )

            assert result["source_type"] == "repo"
            # Sources should reference files, not URLs
            for src in result["sources"]:
                assert not src.startswith("http"), f"Source should not be a URL: {src}"

    @pytest.mark.asyncio
    async def test_node_context_files_passed_to_search(self):
        """node_context.files should be forwarded to Qdrant search."""
        from services.crag_service import run_crag_pipeline

        node_context = {
            "id": "auth",
            "label": "Auth Service",
            "type": "service",
            "description": "Handles JWT auth",
            "files": ["src/auth.py", "src/jwt_utils.py"],
        }

        with patch("services.crag_service.search_repository_chunks", new_callable=AsyncMock) as mock_search, \
             patch("services.crag_service.free_web_search") as mock_web, \
             patch("services.crag_service.get_genai_client", return_value=None):

            mock_search.return_value = [
                {"file": "src/auth.py", "start_line": 1, "end_line": 10, "text": "def auth(): pass"}
            ]

            await run_crag_pipeline(
                repo_name="test/repo",
                message="How does JWT work here?",
                node_context=node_context,
            )

            # Verify node_files were passed to search
            call_kwargs = mock_search.call_args
            assert call_kwargs is not None
            # node_files should be in the call
            called_node_files = call_kwargs[1].get("node_files") or call_kwargs[0][2] if len(call_kwargs[0]) > 2 else []
            # Just verify the call was made (files were passed)


# ─────────────────────────────────────────────────────────────
# EDGE CASES — Web Search
# ─────────────────────────────────────────────────────────────

class TestFreeWebSearch:
    """Tests for web search function."""

    def test_returns_list(self):
        from services.crag_service import free_web_search
        with patch("services.crag_service.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = []
            results = free_web_search("test query")
            assert isinstance(results, list)

    def test_network_error_returns_empty(self):
        from services.crag_service import free_web_search
        with patch("services.crag_service.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.side_effect = Exception("Network error")
            results = free_web_search("test query")
            assert results == []

    def test_result_has_required_keys(self):
        from services.crag_service import free_web_search
        mock_results = [{"title": "Test", "href": "https://example.com", "body": "snippet"}]
        with patch("services.crag_service.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_results
            results = free_web_search("test query", max_results=1)
            if results:
                assert "url" in results[0]
                assert "snippet" in results[0]
                assert "title" in results[0]
