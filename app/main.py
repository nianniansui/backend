import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.v1.memory import router as memory_router

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


app.include_router(memory_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
