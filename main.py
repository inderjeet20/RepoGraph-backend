from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, chat

app = FastAPI(title="RepoGraph API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": "RepoGraph Backend"}
