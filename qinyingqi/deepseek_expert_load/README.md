# DeepSeek 四卡 Expert Load 预实验

这个目录是在 Node0 的 4 张 Ascend 910B1 上运行 DeepSeek MoE、采集 logical
Expert 路由并比较不同 benchmark 负载分布的独立实验。它不会修改或调用
`qinyingqi/expert_load` 中的 GLM5.2 启动、采集和分析脚本。

本地 Mac 不需要也不应该下载模型或 benchmark。本目录中的下载量为零：模型读取
Node0 公共盘，benchmark 直接复用远端已经准备好的 JSONL。

## 1. 模型选择结论

### 1.1 本实验默认目标

默认检查：

```text
/data/node0_disk1/Public/DeepSeek-V4-Flash
```

现有服务器清单记录该目录全部 safetensors 约 148.66 GiB。实测目录有 92 个
shard，但 `model.safetensors.index.json` 只引用其中 46 个；实际加载候选大小应看
audit 的 `active_shard_gib`，不能用全部文件之和估算 HBM。未引用文件暂不删除。

项目锁定的 vLLM-Ascend `v0.22.1rc1` 有
`DeepSeek-V4-Flash-w4a8-mtp` 的 A2 四卡 TP4 + EP 回归测试：

```text
upstream/vllm-ascend/tests/e2e/pull_request/four_card/test_deepseek_v4.py
```

当前公共目录不是上述 ModelSlim checkpoint 的同名副本，而是 DeepSeek-V4 原生
mixed checkpoint。它通过 `config.json` 表达：普通 Linear/Attention 使用 FP8，
MoE Expert 使用 FP4/MXFP4；vLLM-Ascend 会把该 FusedMoE 路径映射到
`w4a8_moe`。因此 audit 同时识别两种合法部署 profile：

- `modelslim_w4a8`：`quant_model_description.json` 明确包含 W4A8；
- `deepseek_v4_native_fp8_fp4`：`quant_method=fp8` 或
  `deepseek_v4_fp8`，且 `expert_dtype=fp4`；该字段缺失时按锁定的 vLLM
  `0.22.1` 行为默认成 `fp4`。

目录名称仍不能证明权重格式，必须先让 `00_audit_model.sh` 证明：

- `model_type=deepseek_v4`；
- 存在 MoE 层、routed Expert 数和 Top-K；
- 上述两个 W4A8 Expert 部署 profile 至少匹配一个；
- safetensors 分片和索引没有缺文件。

任一条件不成立，脚本退出且不会启动容器。

### 1.2 不要用四卡加载 W8A8 目录

```text
/data/node0_disk1/Public/DeepSeek-V4-Flash-w8a8-mtp
```

该目录约 279.41 GiB，已经超过 4 × 64 GiB 的算术 HBM 总量，而且推理还需要
KV cache、workspace 和运行时内存。vLLM-Ascend 教程也要求 8 张 A2，所以不能用
四卡启动。

### 1.3 “最新开源”措辞边界

可从 DeepSeek 官方公开页面严格核验的旗舰是 DeepSeek-V3.2，但它是约 685B 的
MoE；官方/上游 W8A8 部署仍需要 2 台 8×64G A2，四卡不可能完整承载。

本项目上游 `v0.22.1rc1` 已经支持 DeepSeek-V4-Flash，并提供 `gdydems` /
`Eco-Tech` 发布的 W4A8/W8A8 checkpoint。然而在把结果写进对外材料前，还要从
服务器模型目录或下载记录补齐原始发布者、revision、license 和哈希。补齐前建议
使用以下准确描述：

> DeepSeek-V4-Flash W4A8 checkpoint supported by vLLM-Ascend v0.22.1rc1

不要暂时写成“DeepSeek 官方原始权重”。这不影响内部 Expert 路由预实验。

## 2. 与 GLM 实验隔离

本实验使用：

- 独立目录：`qinyingqi/deepseek_expert_load`；
- 独立容器前缀：`deepseek-v4-expert`；
- 独立端口：`127.0.0.1:7100`；
- 独立结果根目录：`/data/node0_disk2/deepseek-expert-load/runs`；
- 独立 current-run-id；
- 只挂载你确认获批的 4 张物理 NPU，并映射为容器内逻辑卡 `0,1,2,3`。

脚本不会停止或启动 keep-alive，也不会停止其他容器。你仍需按照服务器规则手动
停止并恢复本次实际使用的 4 张卡。

## 3. Node0 一次性配置

在 Node0 更新项目后进入目录：

```bash
cd /home/qinyingqi/GLM-VLLM-ASCEND/qinyingqi/deepseek_expert_load

cp configs/node0.env.example configs/node0.env
sed -i 's/\r$//' configs/node0.env scripts/*.sh scripts/*.py
```

确认镜像。优先使用未修改的官方基础镜像：

```bash
docker image inspect quay.io/ascend/vllm-ascend:v0.22.1rc1 \
  --format '{{.Id}} {{.RepoTags}}'
```

如果 Node0 只有之前 GLM 实验生成的派生镜像，也可以将配置里的 `IMAGE_REF` 改为：

```text
glm52-expert-capture:v0.22.1rc1-w8a8-v1
```

它和基础镜像使用相同的 vLLM/vLLM-Ascend 版本；其中 GLM W8A8 hook 不会介入
DeepSeek-V4 W4A8 路径。使用它不会修改 GLM 实验文件。

编辑配置：

```bash
vi configs/node0.env
```

必须填写管理员实际分配给你的四张卡，例如：

```text
HOST_NPU_IDS=0,1,2,3
```

不要照抄示例卡号。其余默认路径应与当前服务器清单一致，但仍应逐项核对。

## 4. 在占卡前审计模型

这个步骤只读 JSON、文件名和文件大小，不加载模型 tensor，也不占 NPU：

```bash
bash scripts/00_audit_model.sh \
  configs/node0.env \
  /tmp/deepseek-v4-model-audit.json
```

成功标志：

```text
"compatible": true
"model_type": "deepseek_v4"
"w4a8_detected": true
"deployment_profile": "deepseek_v4_native_fp8_fp4"
"expert_dtype": "fp4"
"recommended_vllm_quantization": "fp8"
```

`quant_method=deepseek_v4_fp8` 时最后一项会对应输出
`deepseek_v4_fp8`，同样合法。启动脚本会读取该值，不要求手工填写。

同时记录输出中的：

- `num_hidden_layers`；
- `num_experts`；
- `top_k`；
- `moe_layer_indices`；
- `active_shard_gib`；
- `unreferenced_shard_count` 和 `warnings`。

`unreferenced_shard_count` 非零只产生 warning，不会把 compatible 改成 false。
vLLM 会按照 index 过滤未引用 shard；在来源查清前不要移动或删除公共目录文件。

如果新版脚本仍报 `W4A8 Expert execution was not proven`，表示它既不是
ModelSlim W4A8，也不是 DeepSeek-V4 原生 FP8+FP4 mixed profile。不要绕过检查，
也不要在 Mac 下载模型。

## 5. 确认四张卡并启动

先看实时状态和正在暴露 NPU 的容器：

```bash
npu-smi info

docker ps --format '{{.ID}}\t{{.Names}}\t{{.Status}}'

for id in $(docker ps -q); do
  docker inspect "${id}" \
    --format '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}}->{{.PathInContainer}} {{end}}'
done
```

得到管理员确认后，只手动停止这四张卡上的 keep-alive。假设你获批的是
`0,1,2,3`，启动命令为：

```bash
bash scripts/01_launch_tp4.sh \
  configs/node0.env \
  --confirm-npu-ids 0,1,2,3
```

确认参数必须和 `HOST_NPU_IDS` 完全一致。脚本会保存：

```text
${RUN_ROOT}/${RUN_ID}/model-audit.json
${RUN_ROOT}/${RUN_ID}/image-audit.txt
${RUN_ROOT}/${RUN_ID}/npu-smi.before.txt
${RUN_ROOT}/${RUN_ID}/device-smoke.txt
${RUN_ROOT}/${RUN_ID}/launch.command.sh
${RUN_ROOT}/${RUN_ID}/run.env
```

服务配置来自 vLLM-Ascend 四卡测试，但为路由实验做了收敛：

- TP=4、EP 开启；
- 从 `model-audit.json` 自动选择量化参数：当前 mixed checkpoint 使用
  `--quantization fp8`（或配置中原样的 `deepseek_v4_fp8`），ModelSlim checkpoint
  才使用 `--quantization ascend`；
- `max_model_len=8192`；
- `max_num_seqs=1`；
- `--enable-return-routed-experts`；
- eager、关闭 async scheduling、prefix caching、EPLB 和动态负载均衡；
- 不启用 MTP speculative decoding，避免把 draft model 路由混进主模型基线。

加载权重前还会执行四卡 device smoke，成功标志是
`DEVICE_MAPPING_OK`。它只在每张卡分配一个极小 tensor，用来验证物理卡到容器
逻辑卡 `0..3` 的映射。

等待模型加载：

```bash
bash scripts/02_wait_ready.sh configs/node0.env
```

模型加载期间看到 connection refused 是正常的。脚本使用 `--noproxy '*'` 访问
loopback，不会把本地请求发送到服务器代理；最终应输出：

```text
SERVICE_READY
```

另一个终端可以查看状态：

```bash
bash scripts/06_status.sh configs/node0.env
```

## 6. 先做一条路由冒烟

```bash
bash scripts/03_smoke_capture.sh configs/node0.env
```

成功标志：

```text
CAPTURE_OK benchmark=smoke requests=1
```

采集器会动态读取模型拓扑并检查：

- 路由 shape 是 `(prompt_tokens + completion_tokens - 1, layers, top_k)`；
- dense 层路由全为零；
- MoE Expert ID 位于合法范围；
- 每个 token/layer 的 Top-K Expert 不重复；
- 路由不是固定或陈旧的常量。

因此它不会套用 GLM 的 78 层、75 个 MoE 层和 256 Expert 常量。

## 7. 先各跑 5 条 benchmark

使用的输入来自：

```text
/data/node0_disk2/glm52-study/runs/benchmark-data/inputs/
```

先检查文件：

```bash
source configs/node0.env
wc -l "${BENCHMARK_DATA_ROOT}"/inputs/*.jsonl
```

先各跑 5 条，预计很快暴露格式、上下文或路由问题：

```bash
bash scripts/04_run_benchmarks.sh \
  configs/node0.env \
  --benchmarks mmlu_pro,swebench_lite,livecodebench,ruler_niah \
  --max-requests 5
```

这里的“全量”指运行 JSONL 中已经准备好的全部记录。你之前准备数据时使用了
`--limit 50`，因此每个文件是 50 条路由 workload，不是原 benchmark 的完整官方
评测集，也不计算官方 benchmark 分数。

冒烟通过后继续剩余记录，不重复已经成功的请求：

```bash
bash scripts/04_run_benchmarks.sh \
  configs/node0.env \
  --benchmarks mmlu_pro,swebench_lite,livecodebench,ruler_niah \
  --resume
```

每条请求单独保存 request、去掉大体积路由字段后的 response 和 `.npy` 路由；每次
成功后原子更新 aggregate。中断后可以继续 `--resume`。

## 8. 分析 20/90 和 benchmark 差异

5 条预跑结束后就可以先生成一次分析，完整运行后再执行同一命令覆盖派生报表：

```bash
bash scripts/05_analyze.sh configs/node0.env
```

主要输出：

```text
${RUN_ROOT}/${RUN_ID}/analysis/report.md
${RUN_ROOT}/${RUN_ID}/analysis/summary.csv
${RUN_ROOT}/${RUN_ID}/analysis/per-layer.csv
${RUN_ROOT}/${RUN_ID}/analysis/pairwise.csv
${RUN_ROOT}/${RUN_ID}/analysis/analysis.json
```

### 8.1 20% Expert 是否处理 90% token

报告分别计算：

1. 每个 MoE 层各取 `ceil(20% × num_experts)`，然后汇总各层 Top-20% share；
2. 把 `(layer, expert)` 当作独立权重，从全模型选择 20% 最热权重；
3. 达到 90% assignment 实际需要多少 Expert 或 layer-expert 权重；
4. total、prefill、decode 三种阶段分别统计。

HBM/DRAM 放置最应关注前两种。`expert 7` 在不同层是不同权重，不能只把 Expert ID
跨层相加后宣称满足 20/90；报告中的 pooled Expert-ID 只作为诊断指标。

### 8.2 benchmark 是否改变 Expert 分布

`pairwise.csv` 对每两个 benchmark 输出：

- 各层 Jensen-Shannon divergence 的均值和最大值；
- 全部 layer-expert 分布的 JSD；
- Top-20% 热 Expert 集合的逐层平均 Jaccard；
- total、prefill、decode 分阶段结果。

JSD 越大、Top-20% Jaccard 越小，任务类型相关的路由差异越强。最终结论应使用
完整 50 条 workload，并报告 token 数和 bootstrap/重复实验，而不是只看 5 条冒烟。

## 9. 停止服务并恢复占卡

保存日志、停止并删除本实验拥有的容器：

```bash
bash scripts/07_stop.sh configs/node0.env --remove
```

脚本会核对容器的 run-id label，不会按模糊名称停止其他人的容器。最后会打印本次
物理卡号。按照服务器要求，手动恢复且只恢复这四张卡上的 keep-alive。

下一次运行不需要重新下载模型、benchmark 或镜像：重新确认四张卡，执行第 4、5
节即可生成新的 run ID。

## 10. 常见错误

### `W4A8 Expert execution was not proven`

先确认远端已更新新版 `00_audit_model.py`。新版会识别 DeepSeek-V4 原生
FP8+FP4 mixed profile，不要求配置里出现字面量 `W4A8`。如果仍失败，保存 audit
JSON 后确认权重来源；不要换成 279 GiB 的 W8A8 目录。

### audit 显示未引用 shard

这表示目录里存在 `model.safetensors.index.json` 没有引用的额外权重文件。实验记录
使用 `active_shard_gib`；vLLM 加载器会按 index 过滤它们。它不是缺分片错误，暂时
不要清理公共目录。

### 启动日志提示 quantization 不匹配

不要手工把当前 mixed checkpoint 改成 `--quantization ascend`。检查：

```bash
source configs/node0.env
RUN_ID=$(tr -d '[:space:]' < "${RUN_ROOT}/current-run-id")
grep -E 'deployment_profile|recommended_vllm_quantization' \
  "${RUN_ROOT}/${RUN_ID}/model-audit.json"
grep -E 'quantization_profile|vllm_quantization' \
  "${RUN_ROOT}/${RUN_ID}/run.env"
```

二者必须一致；当前模型预期是 native mixed profile 和 `fp8`（或配置原值
`deepseek_v4_fp8`）。

### 镜像没有 `vllm_ascend.models.deepseek_v4`

镜像版本不对。使用 `v0.22.1rc1` 基础镜像或之前相同版本的 GLM 派生镜像。

### 模型加载过程中 connection refused

服务尚未监听端口。继续看 `02_wait_ready.sh` 或 `06_status.sh`，不要重复创建容器。

### 容器早退或 OOM

```bash
source configs/node0.env
RUN_ID=$(tr -d '[:space:]' < "${RUN_ROOT}/current-run-id")
sed -n '1,260p' "${RUN_ROOT}/${RUN_ID}/container.early-exit.log"
```

如果日志不存在：

```bash
bash scripts/06_status.sh configs/node0.env
```

不要直接降低审计门禁或改成 W8A8。先保留完整日志、模型 audit 和实际卡号。

### `routed_experts is missing`

确认启动命令包含 `--enable-return-routed-experts`，并确认镜像是准确的
vLLM/vLLM-Ascend 版本。不要用旧镜像重试。

## 11. 本地与远端测试

Shell/Python 静态检查：

```bash
for script in scripts/*.sh; do bash -n "${script}"; done
python3 -m compileall -q scripts
```

远端 Python 环境已有 NumPy 时运行单元测试：

```bash
python3 -m unittest discover -s tests -v
```

这些测试只构造小型临时 JSON/NPZ，不加载或下载模型。
