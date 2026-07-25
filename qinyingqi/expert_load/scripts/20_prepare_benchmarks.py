#!/usr/bin/env python3
"""Download benchmark prompts on the remote host and emit canonical JSONL inputs.

The emitted records are routing workloads, not official benchmark score inputs.
They preserve task-family prompts for expert-load measurement while avoiding a
dependency on each benchmark's evaluator or sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


BENCHMARKS = (
    "mmlu_pro",
    "swebench_lite",
    "livecodebench",
    "ruler_niah",
)
REMOTE_DATASETS = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "test"),
    "swebench_lite": ("princeton-nlp/SWE-bench_Lite", "test"),
    "livecodebench": ("livecodebench/code_generation_lite", "test"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def first_present(row: dict[str, Any], fields: tuple[str, ...], label: str) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"missing {label}; checked fields: {', '.join(fields)}")


def normalize_options(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("text", value.get("choices", value.get("options")))
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("MMLU-Pro options must be a list with at least two entries")
    options = [as_text(option, "MMLU-Pro option") for option in value]
    if len(options) > 26:
        raise ValueError("MMLU-Pro supports at most 26 answer options in this adapter")
    return options


def stable_request_id(benchmark: str, source_id: str, index: int) -> str:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
    return f"{benchmark}-{index:06d}-{digest}"


def mmlu_pro_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = first_present(row, ("question", "input", "problem"), "MMLU-Pro question")
    options = normalize_options(row.get("options", row.get("choices")))
    choice_lines = "\n".join(
        f"{chr(ord('A') + option_index)}. {option}"
        for option_index, option in enumerate(options)
    )
    source_id = f"{row.get('category', 'unknown')}:{question}"
    metadata = {
        "source_id": source_id,
        "category": row.get("category"),
        "answer": row.get("answer"),
        "answer_index": row.get("answer_index"),
        "option_count": len(options),
    }
    return {
        "request_id": stable_request_id("mmlu_pro", source_id, index),
        "benchmark": "mmlu_pro",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer the following multiple-choice question. Reason carefully, "
                    "then give the answer letter.\n\n"
                    f"Question:\n{question}\n\nChoices:\n{choice_lines}"
                ),
            }
        ],
        "metadata": metadata,
    }


def swebench_lite_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    issue = first_present(row, ("problem_statement", "issue", "prompt"), "SWE-bench issue")
    repo = as_text(row.get("repo"), "SWE-bench repo")
    base_commit = as_text(row.get("base_commit"), "SWE-bench base_commit")
    source_id = as_text(row.get("instance_id"), "SWE-bench instance_id")
    return {
        "request_id": stable_request_id("swebench_lite", source_id, index),
        "benchmark": "swebench_lite",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a software-engineering agent. Analyze the issue, identify "
                    "the relevant code paths, and propose a concrete implementation plan."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Repository: {repo}\nBase commit: {base_commit}\n"
                    f"Issue:\n{issue}\n\n"
                    "Describe the diagnosis and the patch strategy before writing code."
                ),
            },
        ],
        "metadata": {
            "source_id": source_id,
            "repo": repo,
            "base_commit": base_commit,
            "version": row.get("version"),
        },
    }


def livecodebench_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = first_present(
        row,
        ("question_content", "question", "problem", "prompt", "description"),
        "LiveCodeBench problem",
    )
    starter_code = row.get("starter_code", row.get("code", ""))
    if starter_code is None:
        starter_code = ""
    if not isinstance(starter_code, str):
        starter_code = str(starter_code)
    source_id = str(
        row.get("question_id", row.get("problem_id", row.get("id", question)))
    )
    prompt = (
        "Solve this programming problem. Explain the algorithm briefly and provide "
        "correct implementation-ready code.\n\n"
        f"Problem:\n{question}"
    )
    if starter_code.strip():
        prompt += f"\n\nStarter code:\n{starter_code.strip()}"
    return {
        "request_id": stable_request_id("livecodebench", source_id, index),
        "benchmark": "livecodebench",
        "messages": [{"role": "user", "content": prompt}],
        "metadata": {
            "source_id": source_id,
            "contest_date": row.get("contest_date"),
            "difficulty": row.get("difficulty"),
        },
    }


def ruler_niah_record(index: int, filler_words: int, seed: int) -> dict[str, Any]:
    if filler_words < 64:
        raise ValueError("--ruler-words must be at least 64")
    secret = f"GLM52-NIAH-{seed:04d}-{index:04d}"
    filler = (
        "The archive contains routine notes about maintenance schedules, "
        "testing status, and operational observations. "
    ).split()
    before_count = (filler_words * 2) // 3
    words = [filler[position % len(filler)] for position in range(before_count)]
    words.extend(["The", "verification", "key", "is", secret + "."])
    while len(words) < filler_words:
        words.append(filler[len(words) % len(filler)])
    document = " ".join(words)
    source_id = f"seed={seed};index={index};words={filler_words}"
    return {
        "request_id": stable_request_id("ruler_niah", source_id, index),
        "benchmark": "ruler_niah",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read the document and return the verification key exactly.\n\n"
                    f"Document:\n{document}\n\nQuestion: What is the verification key?"
                ),
            }
        ],
        "metadata": {
            "source_id": source_id,
            "needle": secret,
            "filler_words": filler_words,
            "kind": "RULER-style NIAH routing workload, not an official RULER score",
        },
    }


RECORD_BUILDERS: dict[str, Callable[[dict[str, Any], int], dict[str, Any]]] = {
    "mmlu_pro": mmlu_pro_record,
    "swebench_lite": swebench_lite_record,
    "livecodebench": livecodebench_record,
}


def import_dataset_dependencies() -> tuple[Any, Any, Any]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "benchmark preparation requires datasets and huggingface_hub; "
            "install requirements-client.txt in the remote client venv"
        ) from exc
    return load_dataset, HfApi, sys.modules["datasets"]


def load_remote_rows(
    benchmark: str,
    cache_dir: Path,
    requested_revision: str,
    split_override: str | None,
) -> tuple[Iterable[dict[str, Any]], dict[str, str]]:
    load_dataset, hf_api_class, _ = import_dataset_dependencies()
    dataset_id, default_split = REMOTE_DATASETS[benchmark]
    split = split_override or default_split
    api = hf_api_class()
    info = api.dataset_info(repo_id=dataset_id, revision=requested_revision)
    resolved_revision = info.sha
    dataset = load_dataset(
        dataset_id,
        split=split,
        revision=resolved_revision,
        cache_dir=str(cache_dir),
    )
    return dataset, {
        "dataset_id": dataset_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "split": split,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
            output.write(encoded + "\n")
            digest.update((encoded + "\n").encode("utf-8"))
    return digest.hexdigest()


def prepare_remote_benchmark(
    benchmark: str,
    data_root: Path,
    limit: int,
    revision: str,
    split_override: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, source = load_remote_rows(
        benchmark,
        data_root / "hf-cache",
        revision,
        split_override,
    )
    builder = RECORD_BUILDERS[benchmark]
    records: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        records.append(builder(dict(raw_row), row_index))
        if limit and len(records) >= limit:
            break
    if not records:
        raise RuntimeError(f"{benchmark} produced no routing workload records")
    return records, {**source, "selection": "dataset order", "record_count": len(records)}


def prepare_ruler_niah(
    limit: int, filler_words: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record_count = limit if limit else 50
    records = [
        ruler_niah_record(index, filler_words=filler_words, seed=seed)
        for index in range(record_count)
    ]
    return records, {
        "source_type": "deterministic synthetic prompt",
        "record_count": record_count,
        "filler_words": filler_words,
        "seed": seed,
        "kind": "RULER-style NIAH routing workload, not an official RULER score",
    }


def parse_benchmarks(value: str) -> list[str]:
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected:
        raise ValueError("--benchmarks must select at least one benchmark")
    if "all" in selected:
        if len(selected) != 1:
            raise ValueError("--benchmarks=all cannot be combined with explicit benchmarks")
        return list(BENCHMARKS)
    unknown = sorted(set(selected).difference(BENCHMARKS))
    if unknown:
        raise ValueError(f"unsupported benchmark(s): {', '.join(unknown)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        default="mmlu_pro,swebench_lite,livecodebench,ruler_niah",
        help="Comma-separated: mmlu_pro,swebench_lite,livecodebench,ruler_niah,all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Records per workload; use 0 for all remote dataset rows",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Requested Hugging Face revision; the resolved immutable SHA is recorded",
    )
    parser.add_argument(
        "--split",
        help="Override the default split; use only when the selected dataset provides it",
    )
    parser.add_argument("--ruler-words", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    selected = parse_benchmarks(args.benchmarks)
    data_root = args.data_root.expanduser().resolve()
    input_dir = data_root / "inputs"
    manifest_dir = data_root / "manifests"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in selected:
        output_path = input_dir / f"{benchmark}.jsonl"
        manifest_path = manifest_dir / f"{benchmark}.json"
        if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
            raise FileExistsError(
                f"{benchmark} output already exists under {data_root}; use --overwrite "
                "or choose another --data-root"
            )
        if benchmark == "ruler_niah":
            records, source = prepare_ruler_niah(
                args.limit, filler_words=args.ruler_words, seed=args.seed
            )
        else:
            records, source = prepare_remote_benchmark(
                benchmark,
                data_root,
                args.limit,
                args.revision,
                args.split,
            )
        input_sha256 = write_jsonl(output_path, records)
        manifest = {
            "benchmark": benchmark,
            "created_at": utc_now(),
            "input_path": str(output_path),
            "input_sha256": input_sha256,
            "record_count": len(records),
            "source": source,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "records": len(records),
                    "input": str(output_path),
                    "manifest": str(manifest_path),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
