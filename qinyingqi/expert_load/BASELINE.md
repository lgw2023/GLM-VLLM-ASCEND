# Reproducibility Baseline

## Version contract

| Component | Version / commit |
| --- | --- |
| vLLM | `v0.22.1` / `0decac0d96c42b49572498019f0a0e3600f50398` |
| vLLM-Ascend | `v0.22.1rc1` / `5f6faa0cb8830f667266f3b8121cd1383606f2a1` |
| CANN target | `9.0.0` |
| Hardware | 2 x Atlas 800 A2, each 8 x Ascend 910B1 64 GiB |
| Model | `Eco-Tech/GLM-5.2-w8a8` |

`quay.io/ascend/vllm-ascend:glm5.2` is a vendor functional-smoke lane. The formal expert-load lane is a derived image built from the locked group source plus the W8A8 route-capture patch. Never merge results from the two lanes without recording image digest and package versions.

The remote nodes intentionally retain no `.git` metadata. Source provenance is instead content-addressed by `SOURCE_MANIFEST.json`; its SHA-256 is pinned in the byte-identical two-node `cluster.env`, and every preflight, image, HCCL, and launch gate records and revalidates that source ID. The manifest also carries the locked vLLM and vLLM-Ascend commit identifiers used to create the release.

Formal capture also requires an immutable model revision and a derived-image label `glm52.capture_patch_id` matching the run configuration. A configured version string alone is not provenance; the launcher probes installed package metadata before starting.

## Infrastructure gate

- HCCN per-card link/health and L3 ping must pass on both nodes;
- the complete Gitless source manifest and its externally configured SHA-256 must match on both nodes;
- a real 16-rank HCCL `all_reduce` and `all_to_all` must pass for the exact run and image ID;
- image IDs must match across nodes and the indexed model shard set must be complete;
- the operator must confirm that cards 0-7 are available before running step 05.

HCCN ping is a network check and is not evidence that HCCL collectives work.

## Scientific serving contract

- global DP2, local DP1, TP8, EP enabled, PP1;
- real W8A8 weights; dummy/random weight loading is forbidden;
- concurrency 1 and `max-num-seqs=1` for the first route baseline;
- seed 1024, temperature 0, non-streaming requests;
- speculative decoding, async scheduling, prefix cache, graph capture, fused MC2, balance scheduling, dynamic EPLB and redundant experts disabled;
- prefill and decode reported separately;
- warmup/profile requests excluded from measurement.

The locked vLLM auto-enables async scheduling when compatible and defaults prefix caching to enabled. Therefore the capture launcher explicitly uses `--no-async-scheduling` and `--no-enable-prefix-caching`; merely omitting their positive flags is insufficient.

## GLM-5.2 routing contract

- 78 transformer layers total;
- layers 0 through 2 are dense;
- layers 3 through 77 are 75 MoE layers;
- 256 logical routed experts per MoE layer;
- top-8 routed experts per token;
- one shared expert is not counted as a routed expert assignment.

For workload `b`, phase `p`, layer `l`, expert `e`, define `C[b,p,l,e]` as the number of token-expert assignments. The strict 20% budget is `floor(0.2 * 256) = 51` experts; top-52 is reported as rounding sensitivity. Main outputs are:

```text
S20[b,p,l] = share of assignments handled by the 51 hottest experts
K90[b,p,l] = minimum number of experts needed to reach 90% assignments
```

The claim “20% experts handle 90% tokens” is supported only when the assignment-based `K90 <= 51` under the declared aggregation scope. A token that hits any hot expert is not equivalent to eight routed assignments and must not be used as the main statistic.

## Known gate

The OpenAI protocol and `--enable-return-routed-experts` flag exist in vLLM 0.22.1. In the locked Ascend plugin, the common BF16 `FusedMoE.apply` path calls the capturer, but GLM-5.2 W8A8 selects experts inside its quantization method and currently bypasses that call. Consequently, an unpatched W8A8 server can return `null`, empty or zero-filled route data even though the CLI flag is accepted.

No benchmark run is valid until the derived image passes the route gate in `scripts/12_smoke_request.py --require-routes`.
