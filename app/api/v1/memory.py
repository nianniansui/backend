from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pydantic import BaseModel
from uuid import UUID
from datetime import datetime, timezone, timedelta
import logging

from app.db.database import get_db
from app.models.memory import Memory
from app.models.reminder import Reminder
from app.services.ai_service import (
    transcribe_audio,
    embed_text,
    summarize_memory,
    extract_reminder,
)

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


class TextMemoryRequest(BaseModel):
    text: str
    user_id: str = "default"


class ReminderOut(BaseModel):
    id: str
    memory_id: str
    title: str
    trigger_at: str  # ISO8601 with TZ
    created_at: str

    class Config:
        from_attributes = True


class RecapOut(BaseModel):
    """每日摘要：选一条历史记忆推到用户面前"""
    memory_id: str | None
    title: str        # "3 天前你说过：…"
    body: str         # 记忆原文或摘要
    created_at: str | None  # 原记忆的时间


def _parse_uuid(memory_id: str) -> UUID:
    try:
        return UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory id")


def _parse_iso(iso_str: str) -> datetime | None:
    """容错地解析 LLM 返回的 ISO 时间，失败返回 None"""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # LLM 偶尔忘带 tz，认作东八区
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        return dt
    except (ValueError, TypeError):
        logger.warning("could not parse trigger_at: %r", iso_str)
        return None


async def _try_extract_and_save_reminder(
    db: AsyncSession, memory: Memory, raw_text: str
) -> None:
    """录入后异步抽提醒。失败不影响主流程。"""
    try:
        now_iso = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
        result = await extract_reminder(raw_text, now_iso=now_iso)
        if not result:
            return
        trigger_at = _parse_iso(result.get("trigger_at"))
        if trigger_at is None:
            return
        if trigger_at <= datetime.now(timezone.utc):
            return  # 时间已过，不建提醒
        reminder = Reminder(
            user_id=memory.user_id,
            memory_id=memory.id,
            trigger_at=trigger_at,
            title=(result.get("title") or "")[:200],
        )
        db.add(reminder)
        await db.commit()
        logger.info("reminder created for memory %s at %s", memory.id, trigger_at)
    except Exception as e:
        logger.warning("extract_reminder failed (non-fatal): %s", e)


@router.post("/transcribe")
async def transcribe_only(
    audio: UploadFile = File(...),
    user_id: str = Form(default="default"),
):
    """仅做 STT 转写，不存入记忆。用于搜索页语音提问。"""
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    raw_text = await transcribe_audio(audio_bytes, audio.content_type or "audio/wav")
    return {"text": raw_text.strip()}


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

    # 4. 抽提醒（失败不影响主流程）
    await _try_extract_and_save_reminder(db, memory, raw_text)

    return MemoryOut(
        id=str(memory.id),
        raw_text=memory.raw_text,
        summary=memory.summary,
        created_at=memory.created_at.isoformat(),
    )


@router.post("/add_memory_text", response_model=MemoryOut)
async def add_memory_text(
    req: TextMemoryRequest,
    db: AsyncSession = Depends(get_db),
):
    """直接接收文字（来自 Share Extension / 系统分享），跳过 STT。"""
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="文本为空")

    import asyncio
    summary, embedding = await asyncio.gather(
        summarize_memory(raw_text),
        embed_text(raw_text),
    )

    memory = Memory(
        user_id=req.user_id,
        raw_text=raw_text,
        summary=summary,
        embedding=embedding,
    )
    db.add(memory)
    await db.commit()
    await db.refresh(memory)

    await _try_extract_and_save_reminder(db, memory, raw_text)

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


@router.get("/reminders/upcoming", response_model=list[ReminderOut])
async def list_upcoming_reminders(
    user_id: str = "default",
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """返回未来 N 天内待提醒的事件，App 启动时一次性同步并预定到本地通知。"""
    now = datetime.now(timezone.utc)
    until = now + timedelta(days=max(1, min(days, 30)))
    result = await db.execute(
        select(Reminder)
        .where(
            Reminder.user_id == user_id,
            Reminder.trigger_at >= now,
            Reminder.trigger_at <= until,
        )
        .order_by(Reminder.trigger_at.asc())
    )
    rows = result.scalars().all()
    return [
        ReminderOut(
            id=str(r.id),
            memory_id=str(r.memory_id),
            title=r.title,
            trigger_at=r.trigger_at.isoformat(),
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]


@router.get("/recap/today", response_model=RecapOut)
async def recap_today(
    user_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    """每日摘要：从用户历史记忆中挑一条值得回看的，给本地通知用。

    挑选策略（无 LLM 也能跑）：
    1. 1 个月前的今天附近的记忆（"上月今天"）
    2. 1 周前的今天附近的记忆（"上周同一天"）
    3. 一年前今天附近的记忆
    4. 都没有时，挑最早一条（让用户感受到沉淀）
    返回的 title 由前端组装文案，body 是记忆原文/摘要。
    """
    from datetime import date as _date

    # 候选偏移：天数 → 文案前缀
    offsets = [(30, "1 个月前的今天"), (7, "1 周前的今天"), (365, "1 年前的今天")]
    today_utc = datetime.now(timezone.utc)

    for days_back, prefix in offsets:
        target = today_utc - timedelta(days=days_back)
        # 在目标日 ±1 天范围内挑
        lo = target - timedelta(days=1)
        hi = target + timedelta(days=1)
        result = await db.execute(
            select(Memory)
            .where(
                Memory.user_id == user_id,
                Memory.created_at >= lo,
                Memory.created_at <= hi,
            )
            .order_by(Memory.created_at.asc())
            .limit(1)
        )
        m = result.scalar_one_or_none()
        if m:
            body = (m.summary or m.raw_text or "").strip()
            return RecapOut(
                memory_id=str(m.id),
                title=f"{prefix}你说过",
                body=body[:120],
                created_at=m.created_at.isoformat(),
            )

    # 兜底：最早一条
    result = await db.execute(
        select(Memory)
        .where(Memory.user_id == user_id)
        .order_by(Memory.created_at.asc())
        .limit(1)
    )
    m = result.scalar_one_or_none()
    if m:
        body = (m.summary or m.raw_text or "").strip()
        return RecapOut(
            memory_id=str(m.id),
            title="翻翻第一条记忆",
            body=body[:120],
            created_at=m.created_at.isoformat(),
        )

    # 完全没有记录
    return RecapOut(
        memory_id=None,
        title="还没有记忆呢",
        body="按住中间的按钮，随口一记。",
        created_at=None,
    )
