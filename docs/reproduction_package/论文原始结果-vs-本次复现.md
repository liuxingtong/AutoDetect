# AutoDetect 论文原始结果 vs 本次小规模复现

## 摘要结论

本次复现支持论文的**方向性结论**：迭代式自适应搜索后，低分样本比例上升、平均分下降，更容易发现目标模型的弱点。

但本次结果不能当作论文原始结果的严格重复。主要原因是目标模型、Agent 模型、参考答案模型、数据规模和任务范围都不同。本次实验是一个 CPU/API 驱动的 instruction-following 小规模复现，论文则是覆盖 instruction-following、数学和代码的完整实验。

## 论文原始设定与本次设定

| 维度 | 论文原始实验 | 本次复现 | 可比性 |
|---|---|---|---|
| 方法 | Examiner、Questioner、Assessor 协作，并进行迭代搜索 | 实现了同样的三角色流程和迭代搜索 | 方法方向可比 |
| ISR 定义 | 评分低于 4 的回答数 / 总评测数 | 同样使用 `score < 4` | 指标定义可比 |
| 参考/裁判 | 论文迭代搜索使用 GPT-4 参考答案和评分 | 使用 DeepSeek-V4-Flash 负责 Agent、参考答案和裁判 | 不可直接横比 |
| 被测模型 | 多个 Llama、Mistral、GPT-3.5、Claude、GLM 等 | Qwen2.5-7B-Instruct | 不可直接横比 |
| 任务 | instruction-following、mathematics、coding | 仅 instruction-following | 范围不同 |
| 初始问题 | 每个 knowledge point 先生成 5 条 seed questions | 12 个初始知识点，每点 2 条 seed，并追加 3 轮 adaptive | 协议缩小且不同 |
| 规模 | 论文完整实验规模 | 80 条有效样本，约为论文公开 instruction-following 搜索量的 14.4% | 仅方向性参考 |

## 可直接引用的论文结果

论文 v2（2024-12-10）报告了以下结果：

1. 论文将 ISR 定义为低于 4 分的样本比例：

   \[
   ISR = \frac{Num_{<4}}{Num_{total}}
   \]

2. 论文摘要称，AutoDetect 在多个强模型和任务上取得超过 50% 的弱点识别成功率，在 GPT-3.5-turbo 和 Claude-3-sonnet 等模型上也超过 30%。

3. 论文表 1 给出了多个模型在 instruction-following、数学和代码任务上的 ISR。例如：

   | 模型 | Instruction-following Format | Instruction-following General | Instruction-following Overall |
   |---|---:|---:|---:|
   | Llama2-7b-Chat | 55.3% | 37.8% | 43.3% |
   | Mistral-7b-Instruct | 52.9% | 32.5% | 38.2% |
   | Llama3-8b-Instruct | 42.1% | 19.4% | 27.6% |
   | GPT-3.5-turbo | 35.1% | 21.7% | 25.5% |
   | Claude-3-sonnet | 29.3% | 12.8% | 19.2% |

4. 论文表 4 中，方法 `Ours` 在基线比较里的 IFEval 指标为：IFEval-p ISR 56.8%，IFEval-i ISR 42.5%。这两个数字来自论文的完整基线实验，并不是单一模型、单一知识点小实验。

5. 论文文字说明其迭代搜索过程中平均分呈下降趋势，作者将此作为自适应搜索能逐步找到低分问题的证据。

## 本次复现结果

### 总体

| 指标 | 本次复现 |
|---|---:|
| 有效样本 | 80/80 |
| 失败样本 | 0 |
| ISR | 55.00% |
| 平均分 | 4.675 |
| 弱点样本 | 44/80 |
| 评测请求数 | 245 |
| 总 token 数 | 173,782 |

### 迭代阶段

| 阶段 | 样本数 | ISR | 平均分 |
|---|---:|---:|---:|
| Seed | 32 | 40.625% | 5.750 |
| Adaptive 第 1 轮 | 16 | 62.500% | 4.000 |
| Adaptive 第 2 轮 | 16 | 56.250% | 4.375 |
| Adaptive 第 3 轮 | 16 | 75.000% | 3.500 |
| Adaptive 合计 | 48 | 64.583% | 3.958 |

从 Seed 到 Adaptive 合计：

- ISR 从 40.625% 上升到 64.583%，增加 23.958 个百分点；
- 平均分从 5.750 降到 3.958，下降 1.792 分。

这与论文所描述的“迭代搜索平均分下降、后续更容易发现弱点”的方向一致。

## 为什么不能说“复现了论文的 55%”

本次总体 ISR 恰好为 55.00%，与论文中某些表格数字接近，但不能据此声称数值复现成功：

- 被测模型不同；
- 参考答案和裁判模型不同；
- 论文使用 GPT-4 参考答案/评分，本次使用 DeepSeek-V4-Flash；
- 论文使用更完整的 task taxonomy 和更大的搜索量；
- 本次只覆盖 instruction-following，未覆盖数学和代码；
- Seed 数量和迭代调度也进行了缩小调整。

因此，最稳妥的表述是：

> 本次实验在不同模型和缩小协议下，复现了 AutoDetect 的核心趋势，而不是复现论文报告的具体数值。

## 与论文结论的对应关系

| 论文结论 | 本次证据 | 判断 |
|---|---|---|
| 迭代搜索能找到更低分问题 | Adaptive ISR 高于 Seed，Adaptive 平均分低于 Seed | 支持 |
| AutoDetect 可发现模型特定弱点 | 不同 knowledge point 呈现明显不同失败模式和分数 | 初步支持 |
| 方法可跨任务泛化 | 本次没有运行数学和代码任务 | 未验证 |
| 论文完整 ISR 数值可迁移到本次模型 | 模型和裁判均不同 | 不支持直接迁移 |
| 更长运行一定不会发生多样性坍缩 | 本次仅 3 个 adaptive 轮次 | 未验证 |

## 参考来源

- 论文摘要与版本信息：[arXiv:2406.16714](https://arxiv.org/abs/2406.16714)
- 论文 HTML 全文：[AutoDetect v2 HTML](https://arxiv.org/html/2406.16714v2)
- 论文公开代码：[thu-coai/AutoDetect](https://github.com/thu-coai/AutoDetect)

## 本次复现资料

- [中文运行报告](运行报告-中文可读版.md)
- [迭代汇总](迭代汇总-中文可读版.md)
- [逐样本记录](逐样本记录-中文可读版.md)
- [原始样本 JSONL](samples.jsonl)

