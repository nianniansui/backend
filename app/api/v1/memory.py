from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pydantic import BaseModel
from uuid import UUID
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


class MemoryUpdate(BaseModel):
    summary: str | None = None
    raw_text: str | None = None


def _parse_uuid(memory_id: str) -> UUID:
    try:
        return UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory id")


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
    before: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """按时间倒序返回记忆流，支持 `before=<created_at ISO>` 游标分页。"""
    stmt = select(Memory).where(Memory.user_id == user_id)
    if before:
        from datetime import datetime
        try:
            cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid `before` timestamp")
        stmt = stmt.where(Memory.created_at < cutoff)
    stmt = stmt.order_by(Memory.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
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


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    uid = _parse_uuid(memory_id)
    result = await db.execute(
        select(Memory).where(Memory.id == uid, Memory.user_id == user_id)
    )
    mem = result.scalar_one_or_none()
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    await db.delete(mem)
    await db.commit()
    return {"ok": True}


@router.patch("/memories/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    user_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    uid = _parse_uuid(memory_id)
    result = await db.execute(
        select(Memory).where(Memory.id == uid, Memory.user_id == user_id)
    )
    mem = result.scalar_one_or_none()
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    changed = False
    if payload.summary is not None:
        mem.summary = payload.summary.strip()
        changed = True
    if payload.raw_text is not None and payload.raw_text.strip():
        mem.raw_text = payload.raw_text.strip()
        # 改正后的文本需要重算向量，保证搜索准确
        mem.embedding = await embed_text(mem.raw_text)
        changed = True

    if changed:
        await db.commit()
        await db.refresh(mem)

    return MemoryOut(
        id=str(mem.id),
        raw_text=mem.raw_text,
        summary=mem.summary,
        created_at=mem.created_at.isoformat(),
    )


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
