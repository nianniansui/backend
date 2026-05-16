-- 提醒表：从用户的记忆里抽取出来的"将来要发生的事"
CREATE TABLE IF NOT EXISTS reminders (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     VARCHAR(64)   NOT NULL,
    memory_id   UUID          NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    -- 触发时间。LLM 抽出后端转 UTC 存
    trigger_at  TIMESTAMPTZ   NOT NULL,
    -- 给用户看的简短描述，例如 "和王老师 3 点的会"
    title       VARCHAR(200)  NOT NULL,
    -- 是否已经被 App 拉走预定到本地通知（避免重复预定）
    delivered   BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reminders_user_trigger
    ON reminders (user_id, trigger_at);

CREATE INDEX IF NOT EXISTS idx_reminders_user_undelivered
    ON reminders (user_id, delivered, trigger_at);
