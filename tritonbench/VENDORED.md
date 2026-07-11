# Vendored TritonBench

Vendored from [thunlp/tritonbench](https://github.com/thunlp/tritonbench) at commit
`603e28a5050e8c268f6883a69709d477a272d49a` (2025-06-14, "Merge pull request #6 from
thunlp/fix-leakeage"), with SparkDistill-specific modifications on top:

- `tritonbench/` package (runner, validator, evaluator, reporter, CLI, problems) —
  the SparkDistill Triton 3.7.1 / Blackwell benchmark harness (see `PROJECT.md`)
- `bench_config.py`, `configs/`, `UPGRADE_TRITON_3.7.md` — Blackwell runtime gate
  and Triton 3.7.1 pins
- Upstream `EVAL/` scripts adapted for the SparkDistill eval flow

The nested upstream git history was dropped when vendoring; use the commit hash
above to diff against upstream. `LLM_generated/` (upstream's model-output corpus,
~79 MB, referenced by nothing here) is not vendored.

This tree is what `eval.triton_bench` runs to score student checkpoints, and what
SparkProof's release gate reads for decontamination — it must stay in sync between
evaluators and miners, which is why it is committed rather than git-ignored.
