# AutoDetect 自适应探索与分支验证设计

## 1. 目标与范围

本设计在保留 AutoDetect Examiner、Questioner、Target、Evaluator、Assessor 和动态 taxonomy 主流程的基础上，引入四个协作能力：

1. Adaptive Controller：根据搜索历史提出下一轮探索方向并分配候选/测试预算。
2. Novelty Filter：用规则、Embedding 召回和 LLM 语义判断，过滤表面换皮题并保留有实质测试差异的题目。
3. Branch Manager：将失败案例组织为可验证的失败机制分支，记录支持证据、反例和验证状态。
4. Promotion Proposal：在证据规则和 Examiner 语义判断共同通过时，提出新的 Knowledge Point；第一版默认只生成提案，不自动修改正式 taxonomy。

目标不是把 Breadth/Depth 固定成两个题目模板，而是把“探索什么、为何值得探索、证据是否足够”从 Questioner 中分离出来。确定性计算交给 Runtime，开放式语义判断交给 LLM，并通过结构化输出和 checkpoint 保证可追踪。

## 2. 非目标

- 第一版不训练新的策略模型，不引入 Bandit 学习器。
- 不删除 Examiner 的 taxonomy，也不把搜索改成完全开放式生成。
- 不默认自动晋升新的 taxonomy 项。
- 不要求一次性引入远程 embedding 服务；Novelty 层应提供可替换接口，默认实现可采用本地轻量表示或现有依赖。
- 不改变 baseline 模式的既有搜索语义和旧 checkpoint 的读取能力。

## 3. 判断边界

### 3.1 规则/Runtime

负责预算、调用次数、状态流转、字段校验、完全重复、长度/格式约束、正则可验证失败、失败率及停滞统计、checkpoint 原子写入。

### 3.2 Embedding/算法

负责候选与历史题的 Top-K 相似题召回、相似度计算、候选的多样性排序。Embedding 相似不能直接等价为语义重复。

### 3.3 LLM

负责 Controller 的探索目标提议、Questioner 的题目生成、Novelty 的边界语义判断、失败机制假设、候选测试价值判断以及 Branch 是否具有独立测试意义。

### 3.4 混合判断

Novelty 最终决定、失败机制验证和 taxonomy promotion 必须合并规则证据与 LLM 判断。LLM 不能单独确认机制真实存在；一次失败不能直接晋升为 Knowledge Point。

## 4. 组件与接口

### 4.1 新增数据类型

在 `autodetect_repro/types.py` 增加可序列化的数据类型：

- `SearchDecision`：knowledge point、branch、strategy（`exploit|explore|verify|restart`）、objective、reason、candidate/test budget、confidence。
- `CandidateQuestion`：question id、prompt、intent、variation、expected constraints、target mechanism、control variables、changed variables。
- `NoveltyDecision`：question id、相似历史项、substantive-new 标志、relevance score、duplicate type、decision（`keep|reject|uncertain`）和 reason。
- `FailureAnalysis`：failure signature、mechanism hypothesis、observed failure、verified evidence、counterexamples、verification plan、branch status。
- `BranchState`：branch id、所属 taxonomy、支持/反例/验证样本、状态和统计。
- `PromotionProposal`：promotion decision、候选 Knowledge Point、category、description、reason、required followups。

所有新增类型应使用显式字段，避免把未经验证的自由文本混入已有 `SampleRecord` 的事实字段。

### 4.2 Adaptive Controller

新增 `adaptive_controller.py`。接口接收 taxonomy、Branch 状态、近期样本、统计摘要和预算，返回 `SearchDecision`。

第一版提供两个实现：

- `RuleController`：作为可复现 baseline，根据统计指标选择策略。
- `LLMController`：LLM 提议策略与目标，Runtime 校验目标、预算和调用上限。

Controller 不能直接执行调用或修改 taxonomy。预算不足、目标非法或输出无法解析时，使用安全 fallback（均匀分配或 restart），并记录决策失败原因。

### 4.3 Questioner 扩展

保留 `generate_seeds`，将 adaptive 生成扩展为按 Controller 目标一次返回多个 `CandidateQuestion`。Prompt 必须要求每题明确 `intent`、`variation`、`expected_constraints`、`target_failure_mechanism`、`control_variables` 和 `changed_variables`。

Runtime 负责非空、可解析、自包含、候选 id 唯一和题目文本去重；不把 Breadth/Depth 作为强制枚举。

### 4.4 Novelty Filter

新增 `novelty.py`，执行以下固定顺序：

1. 硬规则过滤：空题、超长、格式非法、完全重复。
2. Embedding/表示召回历史 Top-K 近邻。
3. LLM 判断是否只是换实体/场景，或是否改变了实际测试条件、能否服务当前 objective。
4. 规则/MMR 在 test budget 内选择，保留少量随机候选以避免过早收缩。

候选筛选失败时不得静默丢弃：应记录 `uncertain` 或 filter error，并让 Runner 依据策略决定是否保留探索配额。

### 4.5 Evaluator 与失败验证

保留既有分数和弱点判定，同时增加结构化 failure analysis：

- 对 JSON、正则、长度、格式等约束优先使用程序验证；
- 开放式任务由 LLM 辅助评估；
- 证据不足时标记 uncertain/verifying，不强行升级为确定机制；
- 输出中区分 `observed_failure`、`mechanism_hypothesis` 和 `verified_evidence`。

### 4.6 Branch Manager 与 Promotion

新增 `branches.py`。每个候选分支至少记录：支持案例、反例、验证案例、机制假设、当前状态和下一步验证计划。

分支状态为 `candidate → verifying → promoted|rejected`。只有规则证据数量/重复实验/相似度检查通过，且 Examiner 认为它与已有 Knowledge Point 有实质区别并具有独立测试价值，才生成 `promote` 提案。默认只持久化提案；自动加入 taxonomy 由显式配置开关控制。

## 5. Runner 数据流

现有 seed 流程保持不变。adaptive 流程改为：

1. Runner 从 Archive 恢复 taxonomy、Branch 和历史样本。
2. Controller 生成并记录 `SearchDecision`。
3. Questioner 根据决策生成候选题。
4. Novelty Filter 筛选候选并分配 test budget。
5. Target/Evaluator 执行选中的题并产生 `SampleRecord` 与 `FailureAnalysis`。
6. Branch Manager 更新分支状态和验证计划。
7. Assessor/Examiner 生成 `PromotionProposal`。
8. Archive 原子保存样本、决策、筛选结果、分支更新和提案，进入下一轮。

Runner 必须继续支持中断、恢复、最大总样本限制、请求失败记录和旧 `samples.jsonl`。旧 checkpoint 没有新增事件时，应按 baseline 兼容运行。

## 6. 题目与行为多样性测量

多样性采用两层评价，且不预先定义 failure taxonomy：

### 6.1 Semantic Diversity（测试前）

Novelty Filter 对候选题从固定的“历史题库快照”检索 Top-K 近邻，计算：

\[
N_{sem}(q)=1-\frac{1}{k}\sum_{i\in kNN(q,A)}\cos(e_q,e_i)
\]

该值只用于候选排序和轮次统计，不直接等价于行为新颖性。LLM 将候选与近邻判为 `Duplicate`、`Related` 或 `Distinct`：过滤明确 Duplicate，保留 Related/Distinct，并为 Uncertain 保留少量抽样机会。每轮固定使用迭代开始前的历史题库，并对本轮候选内部去重。

每轮记录：

- `semantic_novelty`：本轮新题的平均 `N_sem`；
- `unique_question_gain`：本轮新增非重复有效题数。

Embedding 的平均近邻距离必须通过人工标注题目对进行离线校准；不得把单一最大相似度或该分数本身当作最终重复判定。

### 6.2 Behavioral Diversity（测试后）

Evaluator 将确认失败转换为结构化 Failure Signature，只描述可观察行为和触发条件，不直接声称内部原因：

```json
{
  "violation": "忽略否定约束",
  "observed_behavior": "输出了禁止出现的数字",
  "trigger_context": "多个约束同时存在",
  "evidence": "回答包含2026"
}
```

Behavioral Novelty Analyzer 使用 LLM 将新 Failure Signature 与已有行为簇代表案例归并为 `Same`、`Different` 或 `Uncertain`。Uncertain 不计入已确认的新行为簇；新行为先进入候选簇，不自动升级 taxonomy。行为归并边界需要用小规模人工标注案例校准，避免把同类错误随意拆细。

每轮记录：

- `behavioral_richness`：截至当前轮累计确认的行为簇数 `K_t`；
- `behavioral_gain`：`K_t-K_{t-1}`；
- `discovery_rate`：`behavioral_gain / 本轮 Target 调用次数`。

正式对比 baseline 与改进方法时，须将两者失败案例合并后使用同一离线归并标准，并按首次发现时间回算轮次指标，避免分类口径差异。

## 7. 持久化与兼容性

在现有 `CheckpointStore` 旁增加 append-only 事件文件（建议 `events.jsonl`），事件类型包括 `search_decision`、`candidate_filter`、`failure_analysis`、`behavior_cluster_update`、`diversity_metrics`、`branch_update` 和 `promotion_proposal`。事件内容必须脱敏，不得写入 API key。

manifest schema version 递增；读取旧版本时使用空事件流和默认 branch 状态。事件写入失败不得导致已成功保存的 SampleRecord 被回滚，Runner 应记录可恢复错误并停止当前轮。

## 8. Prompt 合同

`prompts.py` 增加四个结构化 Prompt：

- Controller：输入 taxonomy、branch statistics、recent cases、budget；输出策略、objective、reason、candidate/test budget。
- Questioner：输入当前目标、机制假设、历史案例和预算；输出候选题及变化变量。
- Novelty：输入候选、Top-K 近邻和 objective；输出 substantive-new、相关性、重复类型和 keep/reject/uncertain。
- Failure Signature：输入 Target 回答、参考答案和程序证据；输出可观察的 violation、observed_behavior、trigger_context 和 evidence。
- Behavioral Clustering：输入新 Failure Signature 与已有簇代表案例；输出 Same/Different/Uncertain 及理由。
- Failure/Promotion：输入 Target 回答、参考答案、程序证据、对照案例和历史分支；输出机制假设、验证计划和 promotion decision。

所有 Prompt 必须要求只输出 JSON，并经过现有 structured-call/repair 机制校验。Prompt 不得让 LLM 越权分配预算、直接写入 taxonomy 或把假设描述成事实。

## 9. 配置与运行模式

新增显式配置项：

- `search.mode`: `baseline|adaptive_rule|adaptive_llm`；
- `search.candidate_questions_per_round`；
- `search.novelty_top_k`；
- `search.random_exploration_ratio`；
- `search.auto_promote`（默认 false）；
- 可选的 embedding provider/model 配置。

缺省配置必须保持当前 baseline 行为。adaptive 模式下仍受 `max_total_samples` 和每轮 test budget 双重限制。

## 10. 测试与验收

单元测试至少覆盖：

- 新数据类型 JSON 序列化和旧 checkpoint 恢复；
- Controller 输出解析、非法目标、预算截断和 fallback；
- 硬规则过滤、完全重复、Top-K 召回和 uncertain 保留；
- Semantic Novelty 的平均 kNN 距离、历史快照冻结和 Unique Question Gain；
- Failure Signature 结构化输出、Same/Different/Uncertain 归并和 Behavioral Gain/Discovery Rate；
- branch 状态转换、支持证据/反例计数和 promotion gate；
- Runner 中断、恢复、请求失败和事件追加；
- baseline 模式回归现有测试行为。

离线评估至少比较 `baseline`、`adaptive_rule` 和 `adaptive_llm`：相同 Target 调用预算下的有效失败数、非重复有效失败数、新颖失败发现率、后期停滞程度和失败机制多样性。不得只用 embedding 距离作为“新颖失败”的唯一指标。

## 11. 重新规划的实施顺序

按依赖和研究价值重新排序，并要求每阶段独立提交：

1. **阶段 A：多样性核心与数据契约**
   - 增加 Semantic/Behavioral 指标类型、纯函数测量器、Failure Signature/行为簇数据结构。
   - 先完成离线测试；不接入网络调用。
2. **阶段 B：事件 Archive 与兼容恢复**
   - 扩展事件日志、manifest schema 和旧 checkpoint 读取。
   - 确保指标和行为簇可恢复、可审计。
3. **阶段 C：Semantic Novelty Filter**
   - 实现规则过滤、Embedding Top-K 接口、LLM Duplicate/Related/Distinct 判别和预算选择。
4. **阶段 D：Evaluator Failure Signature 与 Behavioral Analyzer**
   - 增加程序化失败证据、LLM 行为归并、候选簇和每轮 Behavioral Gain/Discovery Rate。
5. **阶段 E：RuleController 与 Runner 集成**
   - 将语义指标、行为指标和停滞统计反馈给规则 Controller；保持 baseline feature flag。
6. **阶段 F：LLMController、Branch/Promotion 与 Prompt 完整接入**
   - LLM 提议探索方向；规则控制预算；默认只生成 promotion proposal。
7. **阶段 G：离线校准与对照实验**
   - 人工标注语义题目对和行为归并案例；统一离线归并两套系统；比较 Semantic Novelty、Unique Question Gain、Behavioral Richness、Behavioral Gain 和 Discovery Rate。

每一步都应先补测试，再实现最小改动；不在本设计范围内引入 Bandit 或大规模重构。
