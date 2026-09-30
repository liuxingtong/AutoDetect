# 代码小样本复现（可读版）

- Agent/参考答案/裁判：`deepseek-ai/DeepSeek-V4-Flash`
- 被测模型：`Qwen/Qwen2.5-7B-Instruct`
- 样本：16
- ISR（score < 4）：43.75%
- 平均分：5.562
- 请求数：49
- 总 token：71382

| 阶段 | 轮次 | 样本 | 弱点 | ISR | 平均分 |
|---|---:|---:|---:|---:|---:|
| seed | 0 | 8 | 3 | 37.50% | 6.250 |
| adaptive | 1 | 4 | 1 | 25.00% | 7.000 |
| adaptive | 2 | 4 | 3 | 75.00% | 2.750 |

## 知识点

- `string_processing / regular_expressions` · seed 第 0 轮 · score=2.0
- `string_processing / regular_expressions` · seed 第 0 轮 · score=2.0
- `string_processing / regular_expressions` · adaptive 第 1 轮 · score=10.0
- `string_processing / regular_expressions` · adaptive 第 2 轮 · score=2.0
- `data_container_operations / filtering_and_selection` · seed 第 0 轮 · score=10.0
- `data_container_operations / filtering_and_selection` · seed 第 0 轮 · score=10.0
- `data_container_operations / filtering_and_selection` · adaptive 第 1 轮 · score=10.0
- `data_container_operations / filtering_and_selection` · adaptive 第 2 轮 · score=4.0
- `logic_and_control_flow / exception_handling` · seed 第 0 轮 · score=2.0
- `logic_and_control_flow / exception_handling` · seed 第 0 轮 · score=5.0
- `logic_and_control_flow / exception_handling` · adaptive 第 1 轮 · score=2.0
- `logic_and_control_flow / exception_handling` · adaptive 第 2 轮 · score=2.0
- `mathematics_and_algorithms / algorithm_design` · seed 第 0 轮 · score=10.0
- `mathematics_and_algorithms / algorithm_design` · seed 第 0 轮 · score=9.0
- `mathematics_and_algorithms / algorithm_design` · adaptive 第 1 轮 · score=6.0
- `mathematics_and_algorithms / algorithm_design` · adaptive 第 2 轮 · score=3.0

原始记录：[`samples.jsonl`](samples.jsonl)，原始汇总：[`iterations.csv`](iterations.csv)。
