import os
import base64
import struct
import logging
import httpx
import dashscope
from dashscope.audio.asr import Transcription
from app.core.config import settings

logger = logging.getLogger(__name__)


# 共享的 httpx client：避免每次请求都做 TCP/TLS 握手。
# 由 lifespan event 管理生命周期；这里只声明，初始化和关闭在 main.py 里。
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """获取共享 httpx client。未初始化时 fallback 到一次性 client。"""
    global _http_client
    if _http_client is None:
        # lifespan 没启动（比如单元测试），用临时 client，性能略差但功能正常
        _http_client = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _http_client


async def init_http_client() -> None:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _parse_wav_header(data: bytes) -> dict:
    """解析 WAV 文件头，返回格式信息"""
    if len(data) < 44 or data[:4] != b'RIFF':
        return {"error": "not a valid WAV", "first4": data[:4].hex()}
    try:
        audio_fmt, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from('<HHIIHH', data, 20)
        return {
            "audio_fmt": audio_fmt,  # 1=PCM, 3=IEEE float
            "channels": channels,
            "sample_rate": sample_rate,
            "bits_per_sample": bits,
            "size": len(data),
        }
    except Exception as e:
        return {"error": str(e)}


async def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """调用阿里云 DashScope 将音频转为文字（文件识别接口）"""
    dashscope.api_key = settings.DASHSCOPE_API_KEY

    fmt = mime_type.split("/")[-1].split(";")[0].strip()
    fmt_map = {"webm": "wav", "mpeg": "mp3", "x-wav": "wav", "x-m4a": "m4a"}
    fmt = fmt_map.get(fmt, fmt)

    if fmt == "wav":
        wav_info = _parse_wav_header(audio_bytes)
        logger.info(f"WAV header: {wav_info}")

    # base64 data URL，无需上传到 OSS 或 Files
    b64 = base64.b64encode(audio_bytes).decode()
    data_url = f"data:audio/{fmt};base64,{b64}"

    trans_resp = Transcription.call(
        model="paraformer-v1",
        file_urls=[data_url],
        api_key=settings.DASHSCOPE_API_KEY,
        language_hints=["zh", "en"],
    )
    logger.info(f"Transcription status={trans_resp.status_code}, output={trans_resp.output}")

    if trans_resp.status_code != 200:
        raise RuntimeError(f"STT failed [{trans_resp.status_code}]: {trans_resp.message}")

    results = trans_resp.output.get("results", [])
    if not results or results[0].get("subtask_status") != "SUCCEEDED":
        logger.warning(f"STT no valid result: {trans_resp.output}")
        return ""

    # paraformer-v1 把转写文本放在 transcription_url 里，需要 fetch
    transcription_url = results[0].get("transcription_url")
    if not transcription_url:
        # 兼容直接返回 sentences 的情况
        sentences = results[0].get("transcription", {}).get("sentences", [])
        text = " ".join(s.get("text", "") for s in sentences)
        logger.info(f"STT text (inline): '{text}'")
        return text

    client = get_http_client()
    r = await client.get(transcription_url, timeout=15)
    r.raise_for_status()
    data = r.json()

    logger.info(f"transcription_url content keys: {list(data.keys())}")
    transcripts = data.get("transcripts", [])
    if not transcripts:
        return ""
    # 优先用顶层 text 字段，fallback 到 sentences 拼接
    text = transcripts[0].get("text", "")
    if not text:
        sentences = transcripts[0].get("sentences", [])
        text = " ".join(s.get("text", "") for s in sentences)
    logger.info(f"STT text: '{text}'")
    return text


async def embed_text(text: str) -> list[float]:
    """调用通义千问 Embedding 将文本向量化"""
    client = get_http_client()
    resp = await client.post(
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
        headers={
            "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.DASHSCOPE_EMBEDDING_MODEL,
            "input": {"texts": [text]},
            "parameters": {"text_type": "document"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["output"]["embeddings"][0]["embedding"]


async def summarize_memory(text: str) -> str:
    """用 DeepSeek 提炼记忆摘要。

    短文本直接返回原文，省一次 LLM 调用（~1s + 一笔 token 费用）。
    阈值 30 字是经验值：典型短句"剪刀放在抽屉里"=8 字，"下周三 3 点和王老师开会"=12 字，
    都不需要再提炼了。
    """
    stripped = text.strip()
    if len(stripped) < 30:
        return stripped

    client = get_http_client()
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
                        "你是一个记忆提炼助手。将用户的口语化记录提炼成简洁的第三人称摘要，"
                        "保留关键实体（物品、地点、人物）和动作，不超过50字。"
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 100,
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


async def extract_reminder(text: str, now_iso: str) -> dict | None:
    """从用户的口语化记忆里抽出"将来要发生的事"。
    返回 {trigger_at: ISO8601, title: str} 或 None（没有时间信息时）。
    `now_iso` 由调用方传入，让 LLM 解算 "明天" / "下周三" 之类相对时间。
    """
    import json

    system = (
        "你从用户记录里抽取需要提醒的未来事件。\n"
        f"当前时间：{now_iso}（ISO8601）。\n"
        "如果记录中包含一个明确指向未来的时间或日期，输出 JSON：\n"
        '{"trigger_at": "YYYY-MM-DDTHH:MM:SS+08:00", "title": "<不超过 30 字、给用户看的事件描述>"}\n'
        "规则：\n"
        "1. 没有任何时间信息（纯日记/物品定位等）时，返回 {\"trigger_at\": null}。\n"
        "2. 时间已过（早于当前时间）时也返回 {\"trigger_at\": null}。\n"
        "3. 只有日期没有时间时，默认设为当天 09:00。\n"
        "4. 模糊时段：'早上'=09:00、'上午'=10:00、'中午'=12:00、'下午'=15:00、'晚上'=20:00。\n"
        "5. 必须返回合法 JSON，不要任何解释。"
    )

    client = get_http_client()
    resp = await client.post(
        f"{settings.LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "max_tokens": 100,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("extract_reminder: bad JSON from LLM: %r", raw)
        return None

    trigger_at = data.get("trigger_at")
    if not trigger_at:
        return None
    title = (data.get("title") or "").strip()
    if not title:
        # title 退化为原文前 30 字，保证下游不空
        title = text.strip().splitlines()[0][:30]
    return {"trigger_at": trigger_at, "title": title}
