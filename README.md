# 小碎 Backend

> 随口一记，随时找回。

FastAPI 后端，提供语音记忆的 STT、向量化、存储与语义搜索能力。

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | FastAPI + Uvicorn |
| 数据库 | PostgreSQL 16 + pgvector (1024-dim) |
| ORM | SQLAlchemy 2.0 (async) |
| STT | 阿里云 DashScope paraformer-v1 |
| Embedding | 阿里云 text-embedding-v3 (1024维) |
| LLM | DeepSeek-V3 (摘要 + RAG 问答) |
| 部署 | Docker Compose (db + api + nginx) |

## 目录结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 入口，CORS，异常处理
│   ├── core/
│   │   └── config.py        # 环境变量配置（pydantic-settings）
│   ├── db/
│   │   └── database.py      # 异步 SQLAlchemy 引擎 + Session
│   ├── models/
│   │   └── memory.py        # Memory ORM 模型
│   ├── api/
│   │   └── v1/
│   │       └── memory.py    # /add_memory, /memories, /search 接口
│   └── services/
│       └── ai_service.py    # STT / Embedding / 摘要 / 搜索
├── migrations/
│   └── 001_init.sql         # 建表 + pgvector 索引
├── nginx.conf               # SPA 路由 + /api/ 反代
├── docker-compose.yml       # db + api + flutter_web 三服务
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 快速启动

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下密钥：

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/xiaosui
DASHSCOPE_API_KEY=your_dashscope_api_key   # 阿里云百炼
DEEPSEEK_API_KEY=your_deepseek_api_key     # DeepSeek
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

### 2. 启动服务

```bash
docker-compose up -d --build
```

三个服务：
- `db` — PostgreSQL 16 + pgvector，端口 `5433`（避免与本地 PG 冲突）
- `api` — FastAPI，端口 `8000`
- `flutter_web` — nginx，端口 `80`，托管 Flutter Web 并反代 API

### 3. 验证

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl "http://localhost:8000/api/v1/memories?user_id=test"
# []
```

## API 接口

### POST `/api/v1/add_memory`

接收音频文件，完成 STT → 摘要 → Embedding → 存储。

```
Content-Type: multipart/form-data
字段：
  audio    File    音频文件（WAV，16kHz 单声道）
  user_id  string  用户 UUID（默认 "default"）
```

响应：
```json
{
  "id": "uuid",
  "raw_text": "我把钥匙放在西安家里了。",
  "summary": "他将钥匙放在西安家中。",
  "created_at": "2026-05-10T13:59:47Z"
}
```

### GET `/api/v1/memories`

```
参数：user_id, limit（默认 20）
```

### POST `/api/v1/search`

语义搜索，返回 AI 回答 + 相关记录。

```json
{
  "query": "钥匙在哪",
  "user_id": "uuid",
  "top_k": 5
}
```

## AI 服务说明

| 服务 | 模型 | 用途 |
|------|------|------|
| DashScope | paraformer-v1 | 语音 → 文字（base64 data URL 方式） |
| DashScope | text-embedding-v3 | 文字 → 1024 维向量 |
| DeepSeek | deepseek-chat | 摘要提炼 + RAG 问答 |

> 如需换用通义千问做 LLM，将 `LLM_BASE_URL` 改为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，`LLM_MODEL` 改为 `qwen-plus`。

## Web 部署（配合前端）

先构建 Flutter Web，再启动完整栈：

```bash
# 在项目根目录
./build_web.sh
```

访问 `http://localhost` 即可使用完整应用。
