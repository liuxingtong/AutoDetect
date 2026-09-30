# 数学小样本复现（可读版）

- Agent/参考答案/裁判：`deepseek-ai/DeepSeek-V4-Flash`
- 被测模型：`Qwen/Qwen2.5-7B-Instruct`
- 样本：16
- ISR（score < 4）：6.25%
- 平均分：9.000
- 请求数：50
- 总 token：45896

| 阶段 | 轮次 | 样本 | 弱点 | ISR | 平均分 |
|---|---:|---:|---:|---:|---:|
| seed | 0 | 8 | 0 | 0.00% | 10.000 |
| adaptive | 1 | 4 | 1 | 25.00% | 7.000 |
| adaptive | 2 | 4 | 0 | 0.00% | 9.000 |

## 知识点

- `algebra / arithmetic / basic_operations` · seed 第 0 轮 · score=10.0
- `algebra / arithmetic / basic_operations` · seed 第 0 轮 · score=10.0
- `algebra / arithmetic / basic_operations` · adaptive 第 1 轮 · score=10.0
- `algebra / arithmetic / basic_operations` · adaptive 第 2 轮 · score=10.0
- `algebra / equations_and_inequalities / quadratic_equations` · seed 第 0 轮 · score=10.0
- `algebra / equations_and_inequalities / quadratic_equations` · seed 第 0 轮 · score=10.0
- `algebra / equations_and_inequalities / quadratic_equations` · adaptive 第 1 轮 · score=1.0
- `algebra / equations_and_inequalities / quadratic_equations` · adaptive 第 2 轮 · score=10.0
- `geometry / trigonometry / trigonometric_identities` · seed 第 0 轮 · score=10.0
- `geometry / trigonometry / trigonometric_identities` · seed 第 0 轮 · score=10.0
- `geometry / trigonometry / trigonometric_identities` · adaptive 第 1 轮 · score=7.0
- `geometry / trigonometry / trigonometric_identities` · adaptive 第 2 轮 · score=6.0
- `analysis / derivatives / chain_rule` · seed 第 0 轮 · score=10.0
- `analysis / derivatives / chain_rule` · seed 第 0 轮 · score=10.0
- `analysis / derivatives / chain_rule` · adaptive 第 1 轮 · score=10.0
- `analysis / derivatives / chain_rule` · adaptive 第 2 轮 · score=10.0

原始记录：[`samples.jsonl`](samples.jsonl)，原始汇总：[`iterations.csv`](iterations.csv)。
