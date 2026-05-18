import logging
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.api.v1.memory import router as memory_router
from app.db.database import engine
from app.services.ai_service import init_http_client, close_http_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XiaoSui API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"422 on {request.method} {request.url.path} | "
                 f"content-type={request.headers.get('content-type')} | "
                 f"body_size={len(body)} | errors={exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.on_event("startup")
async def on_startup():
    """启动时：初始化共享 httpx client + 跑迁移目录里所有 SQL（幂等）。"""
    await init_http_client()
    if os.getenv("SKIP_MIGRATIONS") == "1":
        return
    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    if not migrations_dir.is_dir():
        logger.info("migrations dir not found at %s, skip", migrations_dir)
        return
    files = sorted(p for p in migrations_dir.glob("*.sql"))
    if not files:
        return
    async with engine.begin() as conn:
        for f in files:
            sql = f.read_text(encoding="utf-8")
            logger.info("running migration: %s", f.name)
            try:
                await conn.execute(text(sql))
            except Exception as e:
                # 幂等 SQL 应当用 IF NOT EXISTS；偶发"已存在"错误吞掉就行
                logger.warning("migration %s warning: %s", f.name, e)


@app.on_event("shutdown")
async def on_shutdown():
    await close_http_client()


app.include_router(memory_router)


@app.get("/health")
async def health():
    return {"status": "ok"}

