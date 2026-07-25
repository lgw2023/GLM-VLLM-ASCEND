# Benchmark Routing Workloads

The tracked scripts prepare remote-only prompt workloads and capture expert routes;
they do not replace official benchmark scoring harnesses.

| Workload | Source | Routing purpose |
| --- | --- | --- |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | General knowledge and reasoning |
| `swebench_lite` | `princeton-nlp/SWE-bench_Lite` | Software-engineering issue diagnosis |
| `livecodebench` | `livecodebench/code_generation_lite` | Programming problem solving |
| `ruler_niah` | Deterministic synthetic prompt | RULER-style long-context sensitivity |

`20_prepare_benchmarks.py` resolves each Hugging Face dataset revision to an
immutable SHA and writes canonical JSONL inputs plus a manifest under the remote
data root. `ruler_niah` is intentionally labelled as a routing workload rather
than an official RULER score. tau2-bench and full OpenHands trajectories can use
the same canonical JSONL format once an adapter exports their real multi-turn
messages; do not replace them with fabricated tool traces.

No benchmark data, prompt capture, or route artifact belongs in the source sync.
Use `scripts/22_run_benchmark_suite.sh` only after the W8A8 route gate passes.
