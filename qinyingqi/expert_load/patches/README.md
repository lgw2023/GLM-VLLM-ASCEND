# Route Capture Patch

Phase R003 will place a patch here against exactly:

- vLLM `0decac0d96c42b49572498019f0a0e3600f50398`
- vLLM-Ascend `5f6faa0cb8830f667266f3b8121cd1383606f2a1`

The patch must preserve logical W8A8 `topk_ids` before any logical-to-physical mapping and capture them once in a common MoE layer. It requires unit coverage for BF16 and W8A8 plus an A2 end-to-end route gate. Until that patch and its derived image exist, `vendor_smoke` results are service evidence only, not expert-load evidence.

