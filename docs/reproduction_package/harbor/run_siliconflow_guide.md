# TerminalBench 硅基流动 (SiliconFlow) API 运行指南

> 状态：✅ 已就绪（烟雾测试通过，单任务 trial 未跑）
> 创建时间：2026-10-12
> 关联 issue：10/11 跑 9B 本地模型 + transformers serve 出现 InternalServerError（见 `run_report.md`），改用 35B-A3B + 硅基流动 API 重跑

## 目录

- [为什么换 API](#为什么换-api)
- [文件清单](#文件清单)
- [一次完整运行](#一次完整运行)
- [配置说明](#配置说明)
- [故障排查](#故障排查)
- [回滚到本地 9B](#回滚到本地-9b)

## 为什么换 API

| 维度 | 本地 9B (旧) | 硅基流动 35B-A3B (新) |
|------|--------------|-----------------------|
| 模型 | Qwen3.5-9B 4-bit | Qwen3.5-35B-A3B (MoE, 激活 3B) |
| 推理引擎 | transformers serve (port 8101) | 远端 API |
| 启动耗时 | 冷启 ~5-10 分钟 (PyTorch inductor 编译) | < 1 秒 |
| 单任务超时 | 经常 InternalServerError | 取决于 35B 实际能力 + 网络 |
| 推理稳定性 | 差（cuda:2 抢占、并发冲突） | 取决于 API SLA |
| 成本 | 0 | 按 token 计费（A3B 很便宜） |
| 与 autodetect 主线可比性 | 无（9B ≠ 35B） | **有**（与 autodetect target_model 一致） |

**核心收益**：换 35B 后能跟 `autodetect_repro` 的目标模型对齐，做端到端对比论证。

## 文件清单

```
docs/reproduction_package/harbor/
├── terminalbench-qwen35-9b-bnb4bit-improved.json       # 旧：本地 9B
├── terminalbench-qwen35-9b-bnb4bit-nonthinking-16k.json # 旧：本地 9B
├── terminalbench-qwen35-35b-siliconflow.json           # 新：硅基流动 35B
├── .env.example                                        # 新：API key 模板
├── run_terminalbench_improved.sh                       # 旧：本地 serve
├── run_terminalbench_siliconflow.sh                    # 新：API 入口
├── run_report.md                                       # 旧报告（10/11）
└── run_siliconflow_guide.md                            # 本文档
```

`.env` 文件**不**进 git（`.gitignore` 已覆盖）。

## 一次完整运行

### 前置（只做一次）

```bash
cd docs/reproduction_package/harbor
cp .env.example .env
# 编辑 .env，把 SILICONFLOW_API_KEY=sk-your-real-key-here 替换成你的真实 key
```

### 启动

```bash
bash run_terminalbench_siliconflow.sh
```

脚本会做 4 件事：

1. **加载 .env**，检查 `SILICONFLOW_API_KEY` 存在
2. **scp** 推 config 到远端 `/data/liuxingtong/harbor/jobs/`
3. **远端 curl 烟雾测试**直连 `https://api.siliconflow.cn/v1/chat/completions`（不开 Docker，最快暴露网络/配额问题）
4. **harbor run** 启动完整 TerminalBench

### 看进度

```bash
ssh liuxingtong@115.182.62.246
tail -f /data/liuxingtong/harbor/jobs/terminalbench-qwen35-35b-siliconflow/job.log
```

结果落在 `/data/liuxingtong/harbor/jobs/terminalbench-qwen35-35b-siliconflow/result.json`。

## 配置说明

```json
{
    "model_name": "openai/Qwen/Qwen3.5-35B-A3B",
    "kwargs": {
        "api_base": "https://api.siliconflow.cn/v1",
        "temperature": 0.0,
        "max_turns": 4,
        "enable_summarize": false,
        "interleaved_thinking": false,
        "llm_kwargs": {
            "api_key": "os.environ/SILICONFLOW_API_KEY",
            "timeout": 300
        },
        "llm_call_kwargs": {
            "max_tokens": 8192,
            "top_p": 0.95,
            "timeout": 300,
            "chat_template_kwargs": {
                "enable_thinking": false
            }
        }
    }
}
```

### 关键字段

| 字段 | 值 | 作用 |
|------|----|------|
| `model_name` | `openai/Qwen/Qwen3.5-35B-A3B` | litellm 通过 `openai/` 前缀识别为 OpenAI 兼容协议，转发到硅基流动 |
| `api_base` | `https://api.siliconflow.cn/v1` | 硅基流动 OpenAI 兼容 endpoint |
| `api_key` | `os.environ/SILICONFLOW_API_KEY` | **litellm 原生语法**——运行时从 env 读 key，不写死在 JSON |
| `enable_thinking` | `false` | 关掉 35B 的链式思考（terminalbench 是多轮短任务，开启会浪费 token） |
| `max_turns` | `4` | terminal-bench-2 任务通常 1-3 轮，4 够用 |

### 为什么不直接写明文 key

1. **明文 key 进 git = 永久泄露**——删除只会让历史里仍存在
2. **`os.environ/VAR` 是 litellm 标准语法**（v1.40+），不依赖 harbor 改源码
3. **autodetect 自己的 YAML 也用 `api_key_env` 模式**——保持一致风格
4. **`.gitignore` 已包含 `.env`**——配合 `cp .env.example .env` 工作流

## 故障排查

### 烟雾测试就 401/403

- 检查 `.env` 里 key 是不是还是 `sk-your-real-key-here`
- 检查 key 是否过期：https://cloud.siliconflow.cn/account/ak
- 检查账号余额

### 烟雾测试 200 但 harbor trial 报 InternalServerError

- harbor → litellm 之间的某个环节把 `os.environ/...` 字符串原样发给 API
- **修复**：把 `os.environ/SILICONFLOW_API_KEY` 改成 envsubst 替换。临时方案（不动 commit）：

  ```bash
  # 在远端
  cd /data/liuxingtong/harbor/jobs
  envsubst < terminalbench-qwen35-35b-siliconflow.json > /tmp/sf-resolved.json
  # 把 /tmp/sf-resolved.json 里 api_key 改成真实值再 harbor run
  ```

- **根本方案**：在 `run_terminalbench_siliconflow.sh` 推送前先本地 envsubst，详情见 issue 跟踪

### ⚠️ 已知问题：35B-A3B 默认开 thinking（导致 token 浪费）

**2026-10-12 烟雾测试发现**：

- 极简 prompt `"Reply with exactly one word: ok"` 实际消耗 **170 completion_tokens**
- 其中 **168 tokens 是 `reasoning_content`**（"Thinking Process: 1. Analyze... 2. Evaluate..."）
- 即使我配了 `chat_template_kwargs: {"enable_thinking": false}`，**硅基流动不认这个字段**

**影响**：
- terminal-bench 单任务 4 轮 × 168 reasoning tokens = 672 浪费 token/task
- 全量 80+ 任务预计浪费 ~50k reasoning tokens（按 0.001 元/1k 估算约 0.05 元，影响不大但污染日志）

**未尝试的关闭方法**（按优先级）：
1. 在请求 body 顶层加 `"thinking": false`（Qwen3 官方 API 用法）
2. 改用 `chat_template_kwargs: {"enable_thinking": false}` 之外试试 `chat_template_kwargs: {"thinking": false}` / `"do_thinking": false`
3. 切换到不带动思考的模型（`Qwen/Qwen2.5-32B-Instruct` 或 `Qwen/Qwen3-8B`）—— 见 models list

**修法尚未落地**——下次跑前先查 https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions 看官方推荐的禁用思考参数名。

### trial 跑得慢

- 单 task timeout: 终端可能跑 10-30 分钟属正常
- 整体 terminal-bench-2 全量 ~80+ 个任务，串行预计数小时到一天
- 想中途停：harbor 自带 Ctrl+C，会标记当前 trial 为 cancelled

### 想跑单个任务

把 config 里 `tasks` 改掉：

```json
"tasks": [
    {
        "path": "/data/liuxingtong/official-benchmarks/terminal-bench-2/build-pov-ray"
    }
]
```

## 回滚到本地 9B

```bash
ssh liuxingtong@115.182.62.246
harbor run /data/liuxingtong/harbor/jobs/terminalbench-qwen35-9b-bnb4bit-improved.json
```

但要保证本地 serve 还在（`pgrep -f "transformers serve.*8101"`），否则要先跑 `run_terminalbench_improved.sh` Phase 1。

## 下次跑前 checklist

- [ ] 已在硅基流动控制台作废旧 key 并生成新 key
- [ ] 新 key 已写入远端 `~/.bashrc`（或 `docker run -e SILICONFLOW_API_KEY=...`）
- [ ] 已查 https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions 确认禁 thinking 参数名
- [ ] 已修改 `terminalbench-qwen35-35b-siliconflow.json` 加禁 thinking 字段
- [ ] 已确认 `/data` 至少 50GB 空闲（terminal-bench-2 全量需 ~50GB trial log）

---

文档版本：v1.1（2026-10-12：烟雾测试通过，发现 thinking token 浪费问题）
