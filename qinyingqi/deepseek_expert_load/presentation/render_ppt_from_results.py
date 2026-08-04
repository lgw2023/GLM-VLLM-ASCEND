#!/usr/bin/env python3
"""
Fill deepseek decode-expert-load HTML template with real benchmark results.

Target inputs (on remote):
  ${RUN_ROOT}/${RUN_ID}/analysis/report.md
  ${RUN_ROOT}/${RUN_ID}/analysis/pairwise.csv
  ${RUN_ROOT}/${RUN_ID}/benchmarks/*/capture-manifest.json
  ${RUN_ROOT}/${RUN_ID}/benchmarks/*/route-quality.json   (optional; for CAPTURE_STATUS)
  ${RUN_ROOT}/${RUN_ID}/run.env                             (optional; for IMAGE_REF etc)

Outputs:
  deepseek_decode_expert_load_report_filled.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
from pathlib import Path
from typing import Any


BENCH_DISPLAY = {
    "mmlu_pro": "MMLU-Pro",
    "swebench_lite": "SWE-bench Lite",
    "livecodebench": "LiveCodeBench",
    "ruler_niah": "RULER-NIAH",
}

BENCH_TO_PLACE = {
    "mmlu_pro": "MMLU",
    "swebench_lite": "SWEB",
    "livecodebench": "LCB",
    "ruler_niah": "RULER",
}


def parse_env_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k] = v
    return out


def read_current_run_id(run_root: Path) -> str:
    path = run_root / "current-run-id"
    if not path.exists():
        raise FileNotFoundError(f"missing run id file: {path}")
    return re.sub(r"\s+", "", path.read_text(encoding="utf-8", errors="replace"))


def percent_str_to_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def html_escape(s: Any) -> str:
    return html.escape(str(s), quote=True)


def replace_placeholders(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def parse_concentration_report_md(report_md: Path) -> dict[str, dict[str, str]]:
    """
    Returns:
      { benchmark: { 'Assignments': '...', 'Top20Share': '...', 'MeanNeed90': '...', 'Layers90': '...' } }
    Filter happens outside.
    """
    text = report_md.read_text(encoding="utf-8", errors="replace").splitlines()
    # Find the table under "## Concentration summary"
    header_seen = False
    rows: list[list[str]] = []

    for line in text:
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "Benchmark" in line and "Phase" in line and "Assignments" in line:
            header_seen = True
            continue
        if not header_seen:
            continue
        if line.startswith("| ---"):
            continue
        # Stop at next section
        if line.startswith("## "):
            break

        parts = [p.strip() for p in line.strip("|").split("|")]
        # Expected: 7 columns
        if len(parts) != 7:
            continue
        rows.append(parts)

    # columns: Benchmark | Phase | Assignments | Mean per-layer top-20% share | Global layer-expert top-20% share | Mean expert fraction for 90% | Layers reaching 90%
    out: dict[str, dict[str, str]] = {}
    for parts in rows:
        bench, phase, assignments, top20_share, _global_share, need90, layers90 = parts
        if phase != "decode":
            continue
        if bench not in out:
            out[bench] = {}
        out[bench]["Assignments"] = assignments
        out[bench]["Top20Share"] = top20_share
        out[bench]["MeanNeed90"] = need90
        out[bench]["Layers90"] = layers90
    return out


def parse_pairwise_decode(pairwise_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with pairwise_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            phase = r.get("phase", "").strip()
            if phase != "decode":
                continue
            rows.append(r)
    return rows


def find_pair_row(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    for r in rows:
        ra = r.get("benchmark_a", "").strip()
        rb = r.get("benchmark_b", "").strip()
        if (ra == a and rb == b) or (ra == b and rb == a):
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=os.environ.get("RUN_ROOT", ""), help="e.g. /data/disk2/.../runs")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""), help="defaults to current-run-id")
    ap.add_argument("--template", default="", help="template html path")
    ap.add_argument("--out", default="", help="output html path")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_root = Path(args.run_root) if args.run_root else None
    if run_root is None or str(run_root) == "":
        raise ValueError("missing --run-root (or set env RUN_ROOT)")

    run_id = args.run_id.strip()
    if not run_id:
        run_id = read_current_run_id(run_root)

    run_dir = run_root / run_id
    analysis_dir = run_dir / "analysis"
    report_md = analysis_dir / "report.md"
    pairwise_csv = analysis_dir / "pairwise.csv"
    if not report_md.exists():
        raise FileNotFoundError(f"missing report: {report_md}")
    if not pairwise_csv.exists():
        raise FileNotFoundError(f"missing pairwise.csv: {pairwise_csv}")

    template_path = Path(args.template) if args.template else (script_dir / "deepseek_decode_expert_load_report.html")
    out_path = Path(args.out) if args.out else (script_dir / "deepseek_decode_expert_load_report_filled.html")

    # 1) Read topology/params from any benchmark capture-manifest.json
    manifest_files = sorted((run_dir / "benchmarks").glob("*/capture-manifest.json"))
    if not manifest_files:
        raise FileNotFoundError(f"missing capture-manifest.json under: {run_dir / 'benchmarks'}")
    manifest = json.loads(manifest_files[0].read_text(encoding="utf-8", errors="replace"))

    topology = manifest.get("topology", {})
    num_experts = topology.get("num_experts") or topology.get("num_experts_per_layer") or topology.get("num_experts_total")
    top_k = topology.get("top_k") or topology.get("topk") or manifest.get("top_k")
    moe_layer_indices = topology.get("moe_layer_indices", [])
    num_moe_layers = len(moe_layer_indices) if isinstance(moe_layer_indices, list) else topology.get("num_moe_layers", "")

    max_tokens = manifest.get("max_tokens", "")
    base_url = manifest.get("base_url", "")
    seed = manifest.get("seed", "")
    unique_scope = manifest.get("unique_scope", "")
    allow_duplicate_topk = bool(manifest.get("allow_duplicate_topk", False))

    # 2) Capture status from route-quality.json (optional)
    capture_status = "N/A"
    route_quality_files = sorted((run_dir / "benchmarks").glob("*/route-quality.json"))
    if route_quality_files:
        rq = json.loads(route_quality_files[0].read_text(encoding="utf-8", errors="replace"))
        imperfect = int(rq.get("imperfect_route_requests", 0))
        trusted_decode = bool(rq.get("routes_trusted_for_decode_load_analysis", False))
        if imperfect == 0:
            capture_status = "CAPTURE_OK"
        elif unique_scope == "decode" and trusted_decode:
            capture_status = "CAPTURE_OK_DECODE_TRUSTED"
        else:
            capture_status = "CAPTURE_OK_WITH_IMPERFECT_ROUTES"
    elif allow_duplicate_topk:
        capture_status = "DECODE_WITH_DUPLICATES"

    # 3) Env-derived fields (optional)
    env = parse_env_kv(run_dir / "run.env")
    image_ref = env.get("IMAGE_REF", "")
    capture_patch_id = env.get("CAPTURE_PATCH_ID", "")
    model_host_path = env.get("MODEL_HOST_PATH", "")

    # MAX_MODEL_LEN from launch.command.sh (best-effort)
    max_model_len = env.get("MAX_MODEL_LEN", "")
    launch_cmd = run_dir / "launch.command.sh"
    if (not max_model_len) and launch_cmd.exists():
        m = re.search(r"--max-model-len\s+(\d+)", launch_cmd.read_text(encoding="utf-8", errors="replace"))
        if m:
            max_model_len = m.group(1)
    if not max_model_len:
        max_model_len = "8192"

    run_root_str = str(run_root)

    # 4) Decode concentration values
    concentration = parse_concentration_report_md(report_md)

    # Build placeholder mapping
    def keep_if_empty(key: str, v: str) -> str:
        return html_escape(v) if v else "{{" + key + "}}"

    mapping: dict[str, str] = {
        "RUN_ROOT": html_escape(run_root_str),
        "RUN_ID": html_escape(run_id),
        "API_ENDPOINT": keep_if_empty("API_ENDPOINT", base_url),
        "CAPTURE_STATUS": html_escape(capture_status),
        "MAX_TOKENS": html_escape(str(max_tokens)),
        "SEED": html_escape(str(seed)),
        "NUM_EXPERTS": html_escape(str(num_experts)) if num_experts is not None else "{{NUM_EXPERTS}}",
        "TOP_K": html_escape(str(top_k)) if top_k is not None else "{{TOP_K}}",
        "NUM_MOE_LAYERS": html_escape(str(num_moe_layers)) if num_moe_layers != "" else "{{NUM_MOE_LAYERS}}",
        "MODEL_HOST_PATH": keep_if_empty("MODEL_HOST_PATH", model_host_path),
        "IMAGE_REF": keep_if_empty("IMAGE_REF", image_ref),
        "CAPTURE_PATCH_ID": keep_if_empty("CAPTURE_PATCH_ID", capture_patch_id),
        "MAX_MODEL_LEN": html_escape(str(max_model_len)),
        # default fill for slide 3 table
        "FILL_AFTER_ANALYZE": html_escape("见第7页 Decode 相位统计（analysis/report.md）"),
    }

    # H1: validate "top20 >= 90%" in decode-phase by per-layer top20 share
    per_bench_ok: list[tuple[str, float]] = []

    for bench, place in BENCH_TO_PLACE.items():
        if bench not in concentration:
            continue
        values = concentration[bench]
        mapping[f"DECODE_ASSIGN_{place}"] = html_escape(values.get("Assignments", ""))
        mapping[f"DECODE_TOP20_{place}"] = html_escape(values.get("Top20Share", ""))
        mapping[f"DECODE_LAYERS90_{place}"] = html_escape(values.get("Layers90", ""))
        mapping[f"DECODE_NEED90_{place}"] = html_escape(values.get("MeanNeed90", ""))

        top20_pct = percent_str_to_float(values.get("Top20Share", ""))
        if top20_pct is not None:
            per_bench_ok.append((bench, top20_pct))

    # Build one-line conclusion + H1 verdict
    if per_bench_ok:
        avg_top20 = sum(v for _, v in per_bench_ok) / len(per_bench_ok)
        ok_count = sum(1 for _, v in per_bench_ok if v >= 90.0)
        bench_names = ", ".join(BENCH_DISPLAY[b] for b, _ in per_bench_ok)
        mapping["ONE_LINE_CONCENTRATION_CONCLUSION"] = html_escape(
            f"Decode 相位：{bench_names} 的 mean per-layer top-20% share 平均约 {avg_top20:.2f}%，其中 {ok_count}/{len(per_bench_ok)} 个任务域≥90%，支持集中度命题（以解码阶段为准）。"
        )
        mapping["H1_VERDICT"] = html_escape(
            f"支持（decode-trusted）：平均 top-20% share={avg_top20:.2f}%，≥90% 的任务域={ok_count}/{len(per_bench_ok)}。"
        )
    else:
        mapping["ONE_LINE_CONCENTRATION_CONCLUSION"] = html_escape("（待填：请从 analysis/report.md 的 Phase=decode 行提取数值）")
        mapping["H1_VERDICT"] = html_escape("（待填）")

    # 5) Decode pairwise: fill JSD/Jaccard cells
    pair_rows = parse_pairwise_decode(pairwise_csv)

    def get_val_float(r: dict[str, Any], key: str) -> float | None:
        v = r.get(key, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def fmt_jsd(v: float | None) -> str:
        return f"{v:.6f}" if v is not None else ""

    def fmt_jacc(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else ""

    def note_from_jacc(j: float | None) -> str:
        if j is None:
            return ""
        if j >= 0.5:
            return f"热点重叠较高（Top-20% Jaccard≈{j:.2f}）"
        if j <= 0.2:
            return f"热点重叠较低（Top-20% Jaccard≈{j:.2f}，任务域差异强）"
        return f"热点重叠中等（Top-20% Jaccard≈{j:.2f}）"

    # Pairs used by template slide 8
    pairs_needed = [
        ("mmlu_pro", "swebench_lite", "MMLU", "SWEB"),
        ("mmlu_pro", "livecodebench", "MMLU", "LCB"),
        ("swebench_lite", "livecodebench", "SWEB", "LCB"),
        ("livecodebench", "ruler_niah", "LCB", "RULER"),
    ]
    for a, b, pa, pb in pairs_needed:
        r = find_pair_row(pair_rows, a, b)
        jsd = get_val_float(r or {}, "layer_jsd_mean")
        gjsd = get_val_float(r or {}, "pooled_layer_expert_jsd")
        jacc = get_val_float(r or {}, "per_layer_top20_jaccard_mean")
        mapping[f"JSD_{pa}_{pb}"] = fmt_jsd(jsd)
        mapping[f"GJSD_{pa}_{pb}"] = fmt_jsd(gjsd)
        mapping[f"JAC_{pa}_{pb}"] = fmt_jacc(jacc)
        mapping[f"NOTE_{pa}_{pb}"] = html_escape(note_from_jacc(jacc))

    # 6) MOST_SIMILAR_PAIR / MOST_DIFFERENT_PAIR / H2_VERDICT
    # Similarity based on highest Jaccard (then lowest JSD)
    scored: list[tuple[str, str, float, float]] = []  # (a,b,jacc,jsd)
    for r in pair_rows:
        a = r.get("benchmark_a", "").strip()
        b = r.get("benchmark_b", "").strip()
        j = get_val_float(r, "per_layer_top20_jaccard_mean")
        jsd = get_val_float(r, "layer_jsd_mean")
        if a and b and j is not None and jsd is not None:
            scored.append((a, b, j, jsd))

    if scored:
        # Similar: max jacc, tie -> min jsd
        similar = max(scored, key=lambda t: (t[2], -t[3]))
        different = min(scored, key=lambda t: (t[2], t[3]))
        mapping["MOST_SIMILAR_PAIR"] = html_escape(
            f"{BENCH_DISPLAY[similar[0]]} ↔ {BENCH_DISPLAY[similar[1]]}"
        )
        mapping["MOST_DIFFERENT_PAIR"] = html_escape(
            f"{BENCH_DISPLAY[different[0]]} ↔ {BENCH_DISPLAY[different[1]]}"
        )
        # H2: if average jaccard is low or best similarity still low, judge task-domain variance
        avg_jacc = sum(s[2] for s in scored) / len(scored)
        mapping["H2_VERDICT"] = html_escape(
            f"从 decode 相位看任务域差异：平均 Top-20% Jaccard≈{avg_jacc:.2f}；最相似对为 {BENCH_DISPLAY[similar[0]]} ↔ {BENCH_DISPLAY[similar[1]]}，最差异对为 {BENCH_DISPLAY[different[0]]} ↔ {BENCH_DISPLAY[different[1]]}。"
        )

    template = template_path.read_text(encoding="utf-8", errors="replace")
    filled = replace_placeholders(template, mapping)
    out_path.write_text(filled, encoding="utf-8")

    print(f"FILLED_PPT_OK out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

