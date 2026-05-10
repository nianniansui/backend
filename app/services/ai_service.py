import os
import base64
import struct
import logging
import httpx
import dashscope
from dashscope.audio.asr import Transcription
from app.core.config import settings

logger = logging.getLogger(__name__)


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

    async with httpx.AsyncClient() as client:
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
    async with httpx.AsyncClient() as client:
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
    """用 DeepSeek 提炼记忆摘要"""
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
