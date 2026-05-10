-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE memories (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     VARCHAR(64)  NOT NULL,
    raw_text    TEXT         NOT NULL,
    summary     TEXT,
    embedding   VECTOR(1024),
    audio_url   VARCHAR(512),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ
);

-- 时间倒序查询索引
CREATE INDEX idx_memories_user_created ON memories (user_id, created_at DESC);

-- 向量近似搜索索引（IVFFlat，适合百万级以下）
CREATE INDEX idx_memories_embedding ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
