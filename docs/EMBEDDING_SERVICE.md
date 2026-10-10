# Embedding Service for AutoDetect

本目录包含用于 AutoDetect 框架的 embedding 服务，支持 Qwen3-Embedding-0.6B 等模型。

## 架构概览

```
┌─────────────────────────┐    HTTP API    ┌──────────────────────────────┐
│   AutoDetect Framework  │◄──────────────►│  Embedding Service (Remote) │
│   (本地/其他服务器)      │   http://...:8080   │  - Qwen3-Embedding-0.6B   │
│                         │                │  - V100 GPU (32GB)          │
│   - NoveltyFilter       │                │  - FastAPI + Uvicorn       │
│   - Semantic Diversity  │                └──────────────────────────────┘
└─────────────────────────┘                              ▲
                                                       │
                                              /data/liuxingtong/models/
                                              Qwen3-Embedding-0.6B/
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `embedding_service.py` | Embedding 模型封装，支持本地/远程模型 |
| `embedding_server.py` | FastAPI 服务端，提供 REST API |
| `embedding_client.py` | 客户端，调用远程 embedding 服务 |
| `test_embedding_remote.py` | 集成测试脚本 |

## 部署步骤

### 1. 远程服务器部署

```bash
# SSH 到远程服务器
ssh liuxingtong@115.182.62.246

# 部署代码
cd /data/liuxingtong
mkdir -p autodetect_embedding
# 复制以下文件到该目录:
# - embedding_service.py
# - embedding_server.py
# - embedding_client.py
# - __init__.py

# 启动服务
nohup /data/liuxingtong/envs/llm/bin/python -m autodetect_embedding.embedding_server \
    --model /data/liuxingtong/models/Qwen3-Embedding-0.6B \
    --host 0.0.0.0 \
    --port 8080 \
    --device cuda \
    > /data/liuxingtong/logs/embedding_server.log 2>&1 &

# 验证服务
curl http://localhost:8080/health
```

### 2. 本地使用

```python
from autodetect_repro.embedding_client import get_remote_embedding_function
from autodetect_repro.novelty import NoveltyFilter, NoveltyLabel

# 获取 embedding 函数
embed = get_remote_embedding_function("http://115.182.62.246:8080")

# 使用 NoveltyFilter
filterer = NoveltyFilter(embedding=embed, top_k=3)
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/dimension` | GET | 获取 embedding 维度 |
| `/embeddings` | POST | 批量编码文本 |

### 示例请求

```bash
# 健康检查
curl http://localhost:8080/health

# 获取维度
curl http://localhost:8080/dimension

# 编码文本
curl -X POST http://localhost:8080/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Hello", "World"]}'
```

### 响应格式

```json
{
  "embeddings": [[0.025, -0.104, ...], [0.015, -0.092, ...]],
  "dimension": 1024,
  "count": 2,
  "processing_time_ms": 150.5
}
```

## 管理脚本

```bash
# 启动服务
./scripts/manage_embedding.sh start

# 停止服务
./scripts/manage_embedding.sh stop

# 检查状态
./scripts/manage_embedding.sh status

# 快速测试
./scripts/manage_embedding.sh test

# 运行完整测试
./scripts/manage_embedding.sh tests
```

## 配置示例

在 AutoDetect 配置文件中添加：

```yaml
embedding:
  type: remote
  base_url: http://115.182.62.246:8080
  batch_size: 32
  max_length: 512
  normalize: true
```

## 支持的模型

- **Qwen3-Embedding-0.6B** (1024 维) - 已部署
- **Qwen3-Embedding-2B** - 可扩展

## 性能基准

| 模型 | 维度 | 单条延迟 | 批量 (5条) |
|------|------|----------|------------|
| Qwen3-Embedding-0.6B | 1024 | ~75ms | ~150ms |

## 故障排查

### 服务无法启动
```bash
# 查看日志
tail -f /data/liuxingtong/logs/embedding_server.log

# 检查 GPU 可用性
nvidia-smi
```

### API 返回错误
```bash
# 检查服务状态
curl http://localhost:8080/health

# 重启服务
pkill -f embedding_server
# 然后重新启动
```

## 扩展部署

如需部署其他 embedding 模型：

1. 下载模型到 `/data/liuxingtong/models/`
2. 修改启动命令的 `--model` 参数
3. 重启服务

```bash
# 例如使用 Qwen3-Embedding-2B
nohup /data/liuxingtong/envs/llm/bin/python -m autodetect_embedding.embedding_server \
    --model /data/liuxingtong/models/Qwen3-Embedding-2B \
    --host 0.0.0.0 \
    --port 8080 \
    --device cuda \
    > /data/liuxingtong/logs/embedding_server_2b.log 2>&1 &
```
