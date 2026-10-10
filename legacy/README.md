# legacy/ — original AutoDetect entry points

This directory preserves the **original, per-domain AutoDetect entry
points** as they were published with the paper.  Each subdirectory
is a standalone script that:

- Hard-codes `<API KEY>` and `<MODEL PATH>` placeholders that must
  be edited before running.
- Calls the siliconflow API directly with `requests`, bypassing
  the rest of the `autodetect_repro/` package.
- Writes its result to a local `result/` directory and analyzes it
  with a sibling `count_res.py`.

## Why these are here, and not in the repo root

The reproduction work in this repo is driven by the unified
`autodetect_repro/` package, with a single entry point
(`python -m autodetect_repro run`) that handles all three domains
(IF, math, code) through configuration.  The original scripts in
this directory are kept for:

1. **Reference**: they show what the original paper's code looked
   like, and what the reproduction is trying to faithfully
   reproduce.
2. **Direct numerical reproduction**: if a reviewer needs the
   exact numbers from the paper rather than the small-scale
   reproduction in `docs/reproduction_package/`, these scripts
   are the way to get them.

## What is here

| subdirectory | domain | entry point |
| --- | --- | --- |
| `autodetect_if/` | Instruction-following (IF) | `scripts/autodetect_if.py` |
| `autodetect_math/` | Mathematics | `scripts/autodetect_math.py` |
| `autodetect_code/` | Coding | `scripts/autodetect_code.py` |

Each has a `scripts/<name>.sh` driver that runs the script over
all tasks in the domain, and a `scripts/count_res.py` post-hoc
analyzer that prints the weakness-identification success rate.

## What is NOT here

- No `__init__.py`, no test coverage, no `requirements` for these
  scripts specifically.  They are presented as-is.
- The data files (`<domain>_cat.json`) are checked in; the result
  directories are not.

## How to run (in three steps)

1. Open the script (`legacy/<domain>/scripts/<domain>.py`) and
   replace the `<API KEY>`, `<API URL>`, and `<MODEL PATH>`
   placeholders.
2. From the `legacy/<domain>/scripts/` directory, run
   `bash <domain>.sh`.
3. After the run, run `python count_res.py` to see the weakness
   identification success rate.

## The reproduction in this repo

For the **small-scale reproduction** (12 initial points × 3
adaptive iterations, 80 sample budget, siliconflow-hosted Qwen
target), use the unified package:

```bash
python -m autodetect_repro run --live \
    --config configs/siliconflow_smallscale.yaml
```

See `docs/reproduction_package/` for the config, the per-domain
result reports, and the calibration script.
