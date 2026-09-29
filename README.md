# RepoGraph Backend

FastAPI backend service for RepoGraph with vector search (Qdrant), LLM code analysis (Google Gemini), and web enrichment.

## Features
- **FastAPI** high-performance asynchronous API
- **Qdrant Vector DB** for code chunk embedding storage and similarity search
- **Google Gemini** integration for code intelligence and chat
- **Docker-ready** for one-click deployment on platforms like Render

## Getting Started Locally

### Prerequisites
- Python 3.11+
- Virtual environment

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/inderjeet20/RepoGraph-backend.git
   cd RepoGraph-backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```

5. Run the server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

6. Open interactive API docs:
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

## Docker Deployment (Render / Cloud)

Run with Docker:
```bash
docker build -t repograph-backend .
docker run -p 10000:10000 --env-file .env repograph-backend
```
