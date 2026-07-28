# DeepSeek-V4 Expert Load on Ascend A2

本目录用于采集 DeepSeek-V4 的 logical Expert 路由，对比不同 benchmark 的负载
分布，并验证 20% Expert 是否承担 90% token assignment。它与
`qinyingqi/expert_load` 下的 GLM5.2 双节点实验完全隔离。

Mac 不下载模型和 benchmark。模型、镜像、NPU 计算和结果都留在远端服务器。

## 1. 当前可执行结论

### 1.1 主实验：Node1 八卡 W8A8

在项目逻辑 Node1（`7.150.15.14`）使用：

```text
模型：/data/node0_disk1/Public/DeepSeek-V4-Flash-w8a8-mtp
硬件：8 x Ascend 910B1 64 GiB
并行：DP=1, TP=8, EP=on
量化：ModelSlim W8A8, --quantization ascend
镜像：deepseek-v4-expert-capture:v0.22.1rc1-w8a8-v3
API：http://127.0.0.1:7100/v1
结果：/data/disk2/deepseek-expert-load-w8a8/runs
```

该模型约 279.41 GiB。vLLM-Ascend 官方 DeepSeek-V4-Flash 教程明确支持单台
Atlas 800 A2、8 x 64 GiB 部署 W8A8：

<https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/DeepSeek-V4-Flash.html>

本实验使用官方资源拓扑，但为 Expert 路由基线关闭 MTP、async scheduling、
prefix caching、EPLB 和动态负载均衡；否则可能把额外运行机制混入任务分布差异。
派生镜像在 vLLM-Ascend 通用 `W8A8_DYNAMIC` MoE 执行方法中，调用 Ascend 的
`FusedMoE` routed-expert capturer 来保存 logical Expert ID，并在 DP=1、TP=8 的
prefill 中汇聚 sequence-parallel token 行。启动脚本会同时核验 Docker label 和两个
源码 marker，裸基础镜像或旧 v1/v2 capture 镜像无法通过。

### 1.2 不再在 A2 上启动原生 FP8+MXFP4 目录

```text
/data/node0_disk1/Public/DeepSeek-V4-Flash
```

该目录是原生 mixed checkpoint：Linear/Attention 为 FP8，Expert 为 MXFP4。
在 Ascend 910B1 加载时会调用 `float4_e2m1fn_x2` custom dtype，并报：

```text
RuntimeError: customize_dtype is not supported by the current soc version
```

这是硬件量化格式不兼容，不是 HCCL、卡号、显存或权重缺失。新版 audit 会在
`TARGET_SOC=ASCEND910B1` 时拒绝这个 profile，避免再次占卡后才失败。原 TP4
脚本保留作诊断和其他兼容硬件使用，不作为当前 A2 执行路线。

## 2. Node0 失败运行清理

如果四卡 MXFP4 容器还没删除，在 Node0 执行：

```bash
cd /home/qinyingqi/GLM-VLLM-ASCEND/qinyingqi/deepseek_expert_load
bash scripts/07_stop.sh configs/node0.env --remove
```

然后按照服务器规则，手动恢复且只恢复该次使用的 `4,5,6,7` keep-alive。

## 3. Node1 一次性配置

代码 push 后，在 Node1 更新并进入目录：

```bash
cd /home/qinyingqi/GLM-VLLM-ASCEND
git pull
cd qinyingqi/deepseek_expert_load

source /home/qinyingqi/miniconda3/etc/profile.d/conda.sh
conda activate torch2.6-py3.10
python3 -c 'import numpy; print("NUMPY_OK", numpy.__version__)'

test -f configs/node1_w8a8.env || \
  cp configs/node1_w8a8.env.example configs/node1_w8a8.env
sed -i 's/\r$//' configs/node1_w8a8.env scripts/*.sh scripts/*.py
```

本地配置已被 `.gitignore` 排除。检查以下关键值：

```bash
grep -E '^(IMAGE_REF|CAPTURE_PATCH_ID|MODEL_HOST_PATH|BENCHMARK_DATA_ROOT|RUN_ROOT|HOST_NPU_IDS|TARGET_SOC|REQUIRED_EXPERT_QUANTIZATION)=' \
  configs/node1_w8a8.env
```

应为：

```text
IMAGE_REF=deepseek-v4-expert-capture:v0.22.1rc1-w8a8-v3
CAPTURE_PATCH_ID=deepseek-v4-w8a8-logical-topk-v3
MODEL_HOST_PATH=/data/node0_disk1/Public/DeepSeek-V4-Flash-w8a8-mtp
BENCHMARK_DATA_ROOT=/data/node0_disk2/glm52-study/runs/benchmark-data
RUN_ROOT=/data/disk2/deepseek-expert-load-w8a8/runs
HOST_NPU_IDS=0,1,2,3,4,5,6,7
TARGET_SOC=ASCEND910B1
REQUIRED_EXPERT_QUANTIZATION=w8a8
```

`MODEL_HOST_PATH` 和 `BENCHMARK_DATA_ROOT` 在 Node1 是 Node0 磁盘的网络挂载；
结果写入 Node1 本地 `/data/disk2`。

## 4. Node1 只读预检

先确认路径、磁盘和镜像：

```bash
source configs/node1_w8a8.env
RUN_STORAGE_ROOT="$(dirname "$(dirname "${RUN_ROOT}")")"

hostname
findmnt -T "${MODEL_HOST_PATH}"
findmnt -T "${BENCHMARK_DATA_ROOT}"
findmnt -T "${RUN_STORAGE_ROOT}"
test -r "${MODEL_HOST_PATH}/config.json" && echo MODEL_CONFIG_OK
test -r "${MODEL_HOST_PATH}/quant_model_weights.safetensors.index.json" && \
  echo MODEL_INDEX_OK
du -sh "${MODEL_HOST_PATH}"
df -hT "${MODEL_HOST_PATH}" "${BENCHMARK_DATA_ROOT}" "${RUN_STORAGE_ROOT}"

docker image inspect "${IMAGE_REF}" \
  --format '{{.Id}} {{.RepoTags}} patch={{index .Config.Labels "deepseek.capture_patch_id"}}'
```

检查八张卡以及运行中容器的设备映射：

```bash
npu-smi info

docker ps --format '{{.ID}}\t{{.Names}}\t{{.Status}}'
for id in $(docker ps -q); do
  docker inspect "${id}" \
    --format '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}}->{{.PathInContainer}} {{end}}'
done
```

必须由管理员确认 `0..7` 全部属于本次任务。空容器仅映射 NPU 设备不代表卡在被
使用；以管理员分配、`npu-smi info` 中的进程/显存状态和 keep-alive 状态为准。
不要停止或删除其他人的容器。

### 4.1 构建 DeepSeek v3 route-capture 镜像

首次运行，或此前配置仍然指向任意 v1/v2 image 时，在 Node1 执行：

```bash
bash scripts/07_build_capture_image.sh
```

它只基于本机已有的 `quay.io/ascend/vllm-ascend:v0.22.1rc1` 构建派生镜像；不下载
模型或 benchmark，也不占用 NPU。若基础镜像确实不存在，才在网络正常时显式加
`--confirm-pull-base`。

已有的 `configs/node1_w8a8.env` 不会被 `git pull` 覆盖，因此迁移到 v3 时必须执行：

```bash
sed -i \
  -e 's#^IMAGE_REF=.*#IMAGE_REF=deepseek-v4-expert-capture:v0.22.1rc1-w8a8-v3#' \
  -e 's#^CAPTURE_PATCH_ID=.*#CAPTURE_PATCH_ID=deepseek-v4-w8a8-logical-topk-v3#' \
  configs/node1_w8a8.env

source configs/node1_w8a8.env
docker image inspect "${IMAGE_REF}" \
  --format '{{.RepoTags}} patch={{index .Config.Labels "deepseek.capture_patch_id"}}'
```

最后一行必须显示 `deepseek-v4-w8a8-logical-topk-v3`。不要把旧 image 重命名为
v3：v1 没有正确调用 Ascend capturer，v2 没有汇聚 TP=8 prefill 的其余 token 行。

## 5. 审计 W8A8 模型

这个步骤只读 JSON、索引、文件名和文件大小，不加载 tensor，也不占 NPU：

```bash
bash scripts/00_audit_model.sh \
  configs/node1_w8a8.env \
  /tmp/deepseek-v4-w8a8-audit.json
```

成功标志：

```text
"compatible": true
"model_type": "deepseek_v4"
"deployment_profile": "modelslim_w8a8"
"expert_quantization": "w8a8"
"recommended_vllm_quantization": "ascend"
"soc_compatible": true
```

同时确认 `problems=[]`、`index_name` 为
`quant_model_weights.safetensors.index.json`、索引没有 missing shard，并记录
`active_shard_gib`。官方 `optional/quarot.safetensors` 不在索引中，会作为 warning
列出，但不影响兼容性，也不要删除它。

## 6. 启动 Node1 八卡服务

先按照服务器规则手动停止 `0..7` 的 keep-alive。然后执行：

```bash
bash scripts/08_launch_tp8_w8a8.sh \
  configs/node1_w8a8.env \
  --confirm-npu-ids 0,1,2,3,4,5,6,7
```

脚本在创建服务前依次检查：

1. W8A8 模型、DeepSeek-V4 拓扑和 910B1 兼容门禁；
2. route-capture label/source marker、vLLM `0.22.1`、vLLM-Ascend
   `0.22.1rc1` 和模型实现；
3. 记录运行中容器的 NPU 映射；映射本身只告警，不误判空容器为占卡任务；
4. 宿主机 `0..7` 到容器 `0..7` 的八卡 tensor smoke；
5. API 端口和运行目录不冲突。

成功后输出：

```text
W8A8_MODEL_AUDIT_OK
RUNNING_CONTAINER_NPU_CHECK_OK
DEVICE_MAPPING_OK
LAUNCH_OK
```

如果存在空容器设备映射，第一个标志会改为
`RUNNING_CONTAINER_NPU_MAPPINGS_RECORDED`，这不是失败。若 `npu-smi info` 显示
真实计算进程或管理员未把卡分给你，则不要继续。

运行合同保存在：

```text
${RUN_ROOT}/${RUN_ID}/model-audit.json
${RUN_ROOT}/${RUN_ID}/image-audit.txt
${RUN_ROOT}/${RUN_ID}/npu-smi.before.txt
${RUN_ROOT}/${RUN_ID}/device-smoke.txt
${RUN_ROOT}/${RUN_ID}/launch.command.sh
${RUN_ROOT}/${RUN_ID}/run.env
```

## 7. 等待服务就绪

模型约 279 GiB，并且从网络挂载读取，首次加载可能较慢：

```bash
bash scripts/02_wait_ready.sh configs/node1_w8a8.env
```

最终应输出：

```text
SERVICE_READY
```

另一个终端查看状态和最近日志：

```bash
bash scripts/06_status.sh configs/node1_w8a8.env
```

加载期间 `127.0.0.1:7100` connection refused 是正常状态；容器退出则不是。

## 8. 路由采集 smoke

```bash
bash scripts/03_smoke_capture.sh configs/node1_w8a8.env
```

成功标志：

```text
CAPTURE_OK benchmark=smoke requests=1
```

采集器会从模型配置动态读取 layer、Expert 和 Top-K，并验证：

- 路由 shape 为 `(prompt_tokens + completion_tokens - 1, layers, top_k)`；
- dense 层不包含非零路由；
- Expert ID 范围合法；
- 每个 token/layer 的 Top-K Expert 不重复；
- 返回数据不是固定或陈旧常量。

## 9. 先跑每类 5 条 benchmark

先确认准备好的输入条数：

```bash
source configs/node1_w8a8.env
wc -l "${BENCHMARK_DATA_ROOT}"/inputs/*.jsonl
```

执行小规模预跑：

```bash
bash scripts/04_run_benchmarks.sh \
  configs/node1_w8a8.env \
  --benchmarks mmlu_pro,swebench_lite,livecodebench,ruler_niah \
  --max-requests 5
```

这里运行的是已经准备好的 JSONL workload。之前数据准备使用了 `--limit 50`，
所以每个文件最多 50 条，不是官方 benchmark 全集，也不计算官方任务分数。

## 10. 继续完整 50 条 workload

5 条全部成功后，从已有进度继续：

```bash
bash scripts/04_run_benchmarks.sh \
  configs/node1_w8a8.env \
  --benchmarks mmlu_pro,swebench_lite,livecodebench,ruler_niah \
  --resume
```

每条请求会保存 request、去除大体积路由后的 response 和 `.npy` 路由，并原子更新
aggregate；中断后仍可再次执行 `--resume`。

## 11. 分析 20/90 与 benchmark 差异

```bash
bash scripts/05_analyze.sh configs/node1_w8a8.env
```

主要输出：

```text
${RUN_ROOT}/${RUN_ID}/analysis/report.md
${RUN_ROOT}/${RUN_ID}/analysis/summary.csv
${RUN_ROOT}/${RUN_ID}/analysis/per-layer.csv
${RUN_ROOT}/${RUN_ID}/analysis/pairwise.csv
${RUN_ROOT}/${RUN_ID}/analysis/analysis.json
```

报告分别统计 total、prefill、decode：

- 每层 Top-20% Expert 的 assignment share；
- 全模型 Top-20% `(layer, expert)` 权重的 assignment share；
- 达到 90% assignment 实际需要的 Expert 比例；
- benchmark 两两之间的 Jensen-Shannon divergence；
- 热 Expert 集合的逐层 Jaccard。

`expert 7` 在不同层对应不同权重，不能只按跨层 Expert ID 汇总就宣称满足
20/90。HBM/DRAM 放置应重点看 per-layer 和 `(layer, expert)` 结果。

## 12. 停止并恢复 keep-alive

保存日志、停止并删除本实验拥有的容器：

```bash
bash scripts/07_stop.sh configs/node1_w8a8.env --remove
```

脚本核对 run-id label，不会按模糊名称停止其他容器。完成后按照服务器规则，手动
恢复且只恢复 `0..7` 的 keep-alive。

下次运行不需要重新下载模型、benchmark 或镜像，从第 4 节重新预检即可。

## 13. 常见错误

### `W8A8 Expert execution was not proven`

确认 `MODEL_HOST_PATH` 指向带 `quant_model_description.json` 的
`DeepSeek-V4-Flash-w8a8-mtp`，不是原生 `DeepSeek-V4-Flash` mixed 目录。

### `RUNNING_CONTAINER_NPU_MAPPINGS_RECORDED`

脚本检测到某个运行中容器映射了选中设备。查看：

```bash
source configs/node1_w8a8.env
RUN_ID=$(tr -d '[:space:]' < "${RUN_ROOT}/current-run-id")
cat "${RUN_ROOT}/${RUN_ID}/running-containers.before.json"
```

这不是失败，也不等同于卡正在计算。不要自动停止该容器；结合 `npu-smi info`、
keep-alive 状态和管理员分配确认八卡确实空闲。

### `Docker image is absent`

Node1 尚无目标镜像。先核对现有 tag；不要因为 tag 不同就盲目重新拉取：

```bash
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | sort
```

本实验需要的是已经构建好的
`deepseek-v4-expert-capture:v0.22.1rc1-w8a8-v3`，不是裸基础镜像。先运行
`bash scripts/07_build_capture_image.sh`；不要把旧 v1/v2 镜像重新打 tag，也不要重新
联网下载模型或 benchmark。

### 模型加载超时或容器退出

```bash
bash scripts/06_status.sh configs/node1_w8a8.env
```

或读取保存的退出日志：

```bash
source configs/node1_w8a8.env
RUN_ID=$(tr -d '[:space:]' < "${RUN_ROOT}/current-run-id")
sed -n '1,300p' "${RUN_ROOT}/${RUN_ID}/container.exit.log"
```

### `routed_experts is missing`

确认保存的 `launch.command.sh` 包含 `--enable-return-routed-experts`，并确认镜像内
vLLM/vLLM-Ascend 版本通过 image audit。

### `top-k expert IDs are not unique for every token/layer`

不要继续跑 benchmark，也不要放宽这个校验。对于 DeepSeek-V4 的 `top_k=6`，同一
token/layer 的 6 个 logical Expert ID 必须互异。旧
`glm52-expert-capture:...-v1` 调用了 upstream 的 `router.capture_fn`，但 Ascend
W8A8 路径实际把 capturer 绑定在 `FusedMoE._ascend_routed_experts_capturer`；因此 v1
可能返回未写入的全零 buffer。v2 修正了该 hook，但在 DP=1、TP=8 prefill 中只保存
rank 0 的 sequence-parallel shard，剩余 token 行仍然是全零。v3 会在写 buffer 前通过
TP group all-gather 恢复完整的 logical route tensor。

按第 4.1 节构建 v3 镜像、更新本地 env，然后停止旧服务并新建一个 run：

```bash
bash scripts/07_stop.sh configs/node1_w8a8.env --remove

export RUN_ID="dsv4-w8a8-tp8-v3-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/08_launch_tp8_w8a8.sh \
  configs/node1_w8a8.env \
  --confirm-npu-ids 0,1,2,3,4,5,6,7
bash scripts/02_wait_ready.sh configs/node1_w8a8.env
bash scripts/03_smoke_capture.sh configs/node1_w8a8.env
```

旧 run 中产生的 smoke 或 benchmark route 文件不具备分析价值，不能与 v3 结果混合。

## 14. 本地与远端测试

静态检查：

```bash
for script in scripts/*.sh; do bash -n "${script}"; done
python3 -m compileall -q scripts tests
```

远端 Python 环境带 NumPy 时运行全部测试：

```bash
python3 -m unittest discover -s tests -v
```

测试只构造小型临时 JSON/NPZ，不下载模型，也不占 NPU。
