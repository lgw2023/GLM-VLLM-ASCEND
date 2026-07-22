# Model provenance

The first-stage model contract is pinned to the official ModelScope repository [`Eco-Tech/GLM-5.2-w8a8`](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8/) at commit:

```text
edd93687ef1c3417d0b92e2cd01cf67e9e9c0039
```

Metadata recorded from that commit on 2026-07-22:

| File / property | Expected value |
| --- | --- |
| `config.json` SHA-256 | `817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073` |
| `quant_model_description.json` SHA-256 | `3386f968cd7049fe95f896c1a1aeacaa5c1c0659ac2ed9a42cd783cc48ef29ba` |
| `quant_model_weights.safetensors.index.json` SHA-256 | `dfa97fa50b5e675ff6cea6ddeae3110795b6b7e971e6dc9cf565a4005fcb079c` |
| Indexed safetensors files | 182 (181 quantized weight shards plus `rot.safetensors`) |
| Indexed tensor payload (`metadata.total_size`) | 773,778,904,680 bytes |
| Actual bytes of the 182 indexed files | 773,876,016,944 bytes |
| Architecture / model type | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` |
| MoE contract | 78 layers, first 3 dense, MoE frequency 1, 256 routed + 1 shared expert, top-8 |

`scripts/validate_model_files.py` enforces these metadata hashes, validates the ModelSlim W8A8/W8A8_DYNAMIC description, confines every indexed shard to the model root, checks every safetensors header, and requires both exact payload and file-byte totals. `04_model_manifest.sh --full-sha256` can additionally record every downloaded file's digest for later comparison; same-size payload corruption is outside the fast gate.
