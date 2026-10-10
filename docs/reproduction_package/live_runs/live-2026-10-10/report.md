# AutoDetect Small-Scale Reproduction

## Scope

- Agent/reference/judge substitution: `deepseek-ai/DeepSeek-V4-Flash`
- Target model: `Qwen/Qwen2.5-7B-Instruct`
- Planned initial volume: 12/555 (2.2%) of the released instruction-following search volume
- This is a directional reproduction, not an exact numerical replication.

## Data Quality

- 12 valid samples
- 0 failed samples
- 3 missing initial samples

## Result

- ISR: 41.67%
- Mean score: 5.33
- Adaptive mean score decreased relative to the seed phase.

## Behavioral Diversity

- Behavioral richness: **0**
- Unique question gain: **0**
- Discovery rate: **0.00%**
- Target calls (until report): **0**

## Branch & Promotion

| branch_id | status | supporting_samples |
| --- | --- | --- |
| `excluded words:Use of forbidden word 'but' in the response` | candidate | b11afcc208202eee3be696073727c608164643ab29083efa83317d09ef29043b |
| `word constraint:The response contains exactly 4 words, not the required 5 unique words.` | candidate | af3e5ad4195fdebcb37dd4bdad22dc0e7b5d922dd66c562a03910d6f7a48d1d6 |
| `word constraint:Each word must have a different number of letters` | candidate | beb4986c07ad16a639601368a438e80ea29f5f2f41c2c29d30dd75489cefa06a |

| branch_id | decision | knowledge_point | reason |
| --- | --- | --- | --- |
| `excluded words:Use of forbidden word 'but' in the response` | continue_verification | excluded words | more independent support or controlled verification is required |
| `excluded words:Use of forbidden word 'but' in the response` | continue_verification | excluded words | more independent support or controlled verification is required |
| `excluded words:Use of forbidden word 'but' in the response` | continue_verification | excluded words | more independent support or controlled verification is required |
| `word constraint:The response contains exactly 4 words, not the required 5 unique words.` | continue_verification | word constraint | more independent support or controlled verification is required |
| `excluded words:Use of forbidden word 'but' in the response` | continue_verification | excluded words | more independent support or controlled verification is required |
| `word constraint:The response contains exactly 4 words, not the required 5 unique words.` | continue_verification | word constraint | more independent support or controlled verification is required |
| `excluded words:Use of forbidden word 'but' in the response` | continue_verification | excluded words | more independent support or controlled verification is required |
| `word constraint:The response contains exactly 4 words, not the required 5 unique words.` | continue_verification | word constraint | more independent support or controlled verification is required |
| `word constraint:Each word must have a different number of letters` | continue_verification | word constraint | more independent support or controlled verification is required |

## Search & Filtering

| knowledge_point | strategy | budget (c/t) | iteration | reason |
| --- | --- | --- | --- | --- |
| `specific words` | explore | 2 / 2 | 1 | no confirmed failure signal is available for exploitation |
| `specific words` | explore | 2 / 2 | 2 | no confirmed failure signal is available for exploitation |
| `specific words` | explore | 2 / 2 | 3 | no confirmed failure signal is available for exploitation |
| `excluded words` | explore | 2 / 2 | 1 | no confirmed failure signal is available for exploitation |
| `excluded words` | explore | 2 / 2 | 2 | no confirmed failure signal is available for exploitation |
| `excluded words` | verify | 2 / 2 | 3 | recent failures include behavior not yet sufficiently verified |
| `word constraint` | exploit | 1 / 1 | 1 | existing failures provide a useful local search signal |

_(no candidate_filter events)_

## Behavior Clusters

| label | samples |
| --- | --- |
| uncertain | 3 |

## Recorded Evaluation Usage

- Requests: 36
- Prompt tokens: 4638
- Completion tokens: 6018
- Total tokens: 10656
