from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pydantic import BaseModel
import logging

from app.db.database import get_db
from app.models.memory import Memory
from app.services.ai_service import transcribe_audio, embed_text, summarize_memory

router = APIRouter(prefix="/api/v1", tags=["memory"])
logger = logging.getLogger(__name__)


class MemoryOut(BaseModel):
    id: str
    raw_text: str
    summary: str | None
    created_at: str

    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    top_k: int = 5


class SearchResponse(BaseModel):
    answer: str
    sources: list[MemoryOut]


@router.post("/add_memory", response_model=MemoryOut)
async def add_memory(
    audio: UploadFile = File(...),
    user_id: str = Form(default="default"),
    db: AsyncSession = Depends(get_db),
):
    """
    接收音频文件，完成：STT → 摘要 → Embedding → 存储
    """
    audio_bytes = await audio.read()
    logger.info(f"add_memory: user={user_id}, size={len(audio_bytes)}, content_type={audio.content_type}")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # 1. 语音转文字
    raw_text = await transcribe_audio(audio_bytes, audio.content_type or "audio/wav")
    logger.info(f"STT result: '{raw_text}'")
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="Could not transcribe audio — 录音内容为空，请重试")

    # 2. 并行：摘要 + 向量化（两者互不依赖）
    import asyncio
    summary, embedding = await asyncio.gather(
        summarize_memory(raw_text),
        embed_text(raw_text),
    )

    # 3. 存入数据库
    memory = Memory(
        user_id=user_id,
        raw_text=raw_text,
        summary=summary,
        embedding=embedding,
    )
    db.add(memory)
    await db.commit()
    await db.refresh(memory)

    return MemoryOut(
        id=str(memory.id),
        raw_text=memory.raw_text,
        summary=memory.summary,
        created_at=memory.created_at.isoformat(),
    )


@router.get("/memories", response_model=list[MemoryOut])
async def list_memories(
    user_id: str = "default",
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """按时间倒序返回记忆流"""
    result = await db.execute(
        select(Memory)
        .where(Memory.user_id == user_id)
        .order_by(Memory.created_at.desc())
        .limit(limit)
    )
    memories = result.scalars().all()
    return [
        MemoryOut(
            id=str(m.id),
            raw_text=m.raw_text,
            summary=m.summary,
            created_at=m.created_at.isoformat(),
        )
        for m in memories
    ]


@router.post("/search", response_model=SearchResponse)
async def search_memory(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    """
    语义搜索：Embedding → 向量检索 → 时间重排 → LLM 回答
    """
    import httpx
    from app.core.config import settings

    # 1. 问题向量化
    query_embedding = await embed_text(req.query)

    # 2. 向量相似度检索（cosine，取 top_k * 2 再时间重排）
    fetch_k = req.top_k * 2
    rows = await db.execute(
        text(
            """
            SELECT id, raw_text, summary, created_at,
                   1 - (embedding <=> CAST(:emb AS vector)) AS score
            FROM memories
            WHERE user_id = :uid
            ORDER BY embedding <=> CAST(:emb AS vector)
            LIMIT :k
            """
        ),
        {"emb": str(query_embedding), "uid": req.user_id, "k": fetch_k},
    )
    candidates = rows.fetchall()

    if not candidates:
        return SearchResponse(answer="没有找到相关记录。", sources=[])

    # 3. 按时间倒序重排，取 top_k
    sorted_candidates = sorted(candidates, key=lambda r: r.created_at, reverse=True)[: req.top_k]

    # 4. 构建上下文交给 LLM
    context = "\n".join(
        f"[{r.created_at.strftime('%Y-%m-%d %H:%M')}] {r.summary or r.raw_text}"
        for r in sorted_candidates
    )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.LLM_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是用户的第二记忆助手。根据以下按时间倒序排列的记录，"
                            "回答用户的问题。优先使用最新的记录，并在回答中注明时间。"
                            "如果记录中没有相关信息，直接说不知道。\n\n"
                            f"记录：\n{context}"
                        ),
                    },
                    {"role": "user", "content": req.query},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()

    sources = [
        MemoryOut(
            id=str(r.id),
            raw_text=r.raw_text,
            summary=r.summary,
            created_at=r.created_at.isoformat(),
        )
        for r in sorted_candidates
    ]
    return SearchResponse(answer=answer, sources=sources)
