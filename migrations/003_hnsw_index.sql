-- 把向量索引从 IVFFlat 切换到 HNSW
-- IVFFlat 需要数据量训练 lists，小表反而比顺序扫慢；HNSW 在小表也快，大表更准。
-- 幂等：先 DROP IF EXISTS 再 CREATE。

DROP INDEX IF EXISTS idx_memories_embedding;

CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
