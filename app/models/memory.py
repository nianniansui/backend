import uuid
from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from app.db.database import Base


class Memory(Base):
    __tablename__ = "memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(64), nullable=False, index=True)
    raw_text = Column(Text, nullable=False)          # STT 原始转写文本
    summary = Column(Text, nullable=True)            # LLM 提炼的摘要
    embedding = Column(Vector(1024), nullable=True)  # text-embedding-v3 维度
    audio_url = Column(String(512), nullable=True)   # 原始音频存储路径（可选）
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
