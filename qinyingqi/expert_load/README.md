# GLM-5.2 expert-load on Ascend A2

本目录用于在两台 Atlas 800 A2（每台 8 张 Ascend NPU）上启动
`Eco-Tech/GLM-5.2-w8a8`，验证双节点 HCCL，并为后续 expert 路由采集和
benchmark 实验提供运行入口。

所有远端命令都使用 Ascend 原生组件：`npu-smi`、`hccn_tool`、HCCL、
`torch-npu` 和 `/dev/davinci*`。不要替换成 CUDA、NCCL 或 NVIDIA 命令。

服务器源码已清空、模型已经下载好的完整双节点流程见
[`REMOTE_A2_FROM_SCRATCH.md`](REMOTE_A2_FROM_SCRATCH.md)。

## 运行原则

- 服务器上的 NPU 占用由操作者手工管理，本项目不会调用服务器维护脚本。
- 执行步骤 05 前，操作者自行用 `npu-smi info` 确认 0–7 卡可用。
- 旧调试轮次产生的本地状态目录不再参与启动，也不需要删除。
- 模型、镜像和 benchmark 数据只放远端服务器，不下载到 Mac。
- 两台节点必须使用相同 `RUN_ID`、相同镜像 ID 和相同模型路径。

## 1. 直接同步源码目录

源码不再做 SHA、manifest 或 bundle 校验，也不需要在服务器解压发布包。服务器
禁止保存 `.git` 不影响直接复制这个子目录；使用团队允许的 `rsync`、`scp -r`
或文件传输工具，把 Mac 上的 `qinyingqi/expert_load/` 同步到两节点即可。

不要同步模型、benchmark 数据、运行日志和真实配置。远端只做最小存在性检查：

```bash
cd <远端源码目录>/qinyingqi/expert_load
test -x scripts/00_preflight.sh
test -x scripts/10_launch_node.sh
for script in scripts/00_preflight.sh scripts/10_launch_node.sh; do
  bash -n "${script}"
done
```

不再需要 `SOURCE_MANIFEST_SHA256`。旧 `cluster.env` 中即使暂时保留这一行，
新脚本也会忽略它；为了清晰可以手工删掉。

## 2. 配置两节点

```bash
cp configs/cluster.env.example configs/cluster.env
cp configs/node0.env.example configs/node.env  # node0
cp configs/node1.env.example configs/node.env  # node1
cp configs/remote_npu_ips.txt.example configs/remote_npu_ips.txt
```

两节点 `cluster.env` 必须一致。`NODE0_COORDINATOR_IP`、`LOCAL_IP`、
`PEER_IP` 和 `LOCAL_NIC` 必须使用承载 HCCL/Gloo 的通信网络地址，不能直接
假设 SSH 管理地址可用。API 与分布式通信是两条不同的路径：
`API_BIND_HOST=127.0.0.1` 只用于 node0 的本地 OpenAI API，不能替代
`NODE0_COORDINATOR_IP`。

`node.env` 只保留节点本身配置：

```bash
NODE_RANK=0或1
LOCAL_IP=<本节点通信IP>
PEER_IP=<另一节点通信IP>
LOCAL_NIC=<通信网卡名>
AUTHORIZED_NPU_IDS=0,1,2,3,4,5,6,7
MODEL_HOST_PATH=/data/node0_disk2/glm52-study/models/GLM-5.2-w8a8
RUN_HOST_ROOT=/data/node0_disk2/glm52-study/runs
LOCAL_STATE_ROOT=<本节点数据盘>/glm52-study/local-state
```

真实配置在 `.gitignore` 中，不要提交。

## 3. 两节点预检和 HCCN

每次源码或配置变化后，两节点分别执行：

```bash
bash scripts/00_preflight.sh configs/cluster.env configs/node.env
bash scripts/01_hccn_ping.sh \
  configs/cluster.env configs/node.env configs/remote_npu_ips.txt
```

## 4. 镜像和模型门禁

镜像已经通过 `docker load` 导入时，两节点执行：

```bash
bash scripts/02_pull_image.sh \
  configs/cluster.env configs/node.env --confirm-existing-image
```

模型已经下载完成后，在 node0 执行：

```bash
bash scripts/03_download_model.sh \
  configs/cluster.env configs/node.env --adopt-existing
bash scripts/04_model_manifest.sh configs/cluster.env configs/node.env
```

`--adopt-existing` 只读取并严格校验现有模型，然后创建 revision-bound 状态记录；
不会联网或重新下载权重。node1 使用相同的共享模型路径，不执行 03、不重新下载。

## 5. 创建本轮 RUN_ID

只在 node0 生成一次：

```bash
source configs/cluster.env
source configs/node.env
mkdir -p "${RUN_HOST_ROOT}/operator"
export RUN_ID="vendor-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "${RUN_ID}" > "${RUN_HOST_ROOT}/operator/current-run-id"
```

node1 读取相同值：

```bash
source configs/cluster.env
source configs/node.env
export RUN_ID="$(tr -d '[:space:]' < "${RUN_HOST_ROOT}/operator/current-run-id")"
```

两个终端都打印确认：

```bash
printf 'node=%s run_id=%s\n' "${NODE_RANK}" "${RUN_ID}"
```

## 6. 两节点记录 NPU 状态

先由操作者完成服务器要求的占卡进程处理，并确认卡可用。随后两节点执行：

```bash
npu-smi info
bash scripts/05_prepare_npus.sh configs/cluster.env configs/node.env
```

成功标志是 `NPU_READY`。该脚本不会启动、停止或恢复任何服务器维护进程；
发现其他暴露 NPU 设备的 Docker 容器时只给出警告。

## 7. 两节点 HCCL 集合通信

先在 node0 启动，随后立即在 node1 启动同一命令：

```bash
bash scripts/06_hccl_collective.sh configs/cluster.env configs/node.env
```

两边都必须出现 `HCCL_COLLECTIVE_OK`。

## 8. 启动 GLM-5.2

先 node0、随后立即 node1：

```bash
bash scripts/10_launch_node.sh configs/cluster.env configs/node.env
```

node0 等待 API：

```bash
bash scripts/11_wait_ready.sh configs/cluster.env configs/node.env
```

看到 `SERVICE_READY` 后，先做最小生成验证：

```bash
python3 -m venv .client-venv
.client-venv/bin/pip install -r requirements-client.txt

.client-venv/bin/python scripts/12_smoke_request.py \
  --base-url "http://${API_BIND_HOST:-127.0.0.1}:${API_PORT}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --output-dir "${RUN_HOST_ROOT}/${RUN_ID}/client-smoke"
```

`vendor_smoke` 只证明模型成功加载和生成，不是 expert-load 结果。

## 9. 停止本项目启动的模型容器

需要结束本轮服务时，在 node0、node1 分别执行：

```bash
bash scripts/19_stop_node.sh configs/cluster.env configs/node.env --remove
```

该脚本只操作带有本轮 ownership label 和精确 container ID 的 GLM 容器，不会
操作其他用户容器或服务器维护进程。

## 10. 正式 expert capture 和 benchmark

`vendor_smoke` 只能证明模型服务可用，不能产生正式 expert-load 结果。批量
脚本会拒绝 `vendor_smoke`，也会在任何一个请求缺少真实 `routed_experts` 时停止。
这不是额外的部署门槛，而是避免把空数组、零数组或文本输出误当作 expert 路由。

### 10.1 先准备 route-capture 派生镜像

正式实验使用的不是之前已经停止的 `vendor_smoke` 容器。官方
`quay.io/ascend/vllm-ascend:v0.22.1rc1` 已包含 vLLM 的 route-return 框架，但其
W8A8 路径绕过了已绑定的 router capture callback。本项目的派生镜像只在
`w8a8_dynamic.py` 的 logical `topk_ids` 选择后插入一次 callback；不复制模型，
不修改模型权重。

先在能访问 Quay 的 **node1** 运行。node1 若已经有该 base image，可以去掉
`--confirm-pull-base`：

```bash
cd "${PROJECT_ROOT}/qinyingqi/expert_load"
bash scripts/07_build_capture_image.sh --confirm-pull-base
```

若这里报 `Client.Timeout exceeded while awaiting headers`，这是 Docker daemon 到
Quay 超时，并非补丁或 NPU 错误，也不是 Quay 401 认证问题。先检查精确 base tag
是否已经存在：

```bash
docker image inspect quay.io/ascend/vllm-ascend:v0.22.1rc1 \
  --format '{{.Id}}'
```

如果能输出 image ID，直接离线构建；更新后的脚本即使带了 pull flag 也会跳过 pull：

```bash
bash scripts/07_build_capture_image.sh
```

如果 inspect 失败，node1 确实缺少 base。`quay.io/ascend/vllm-ascend:glm5.2`
是 vLLM 0.21 的 vendor smoke image，不能冒充该 base。此时需要管理员恢复 Docker
daemon 的 registry 网络/镜像代理，或在另一台能拉 Quay 的 **Linux 服务器**上对
精确 tag 执行 `docker pull`、`docker save`，传到 node1 后 `docker load`。不要在 Mac
拉这个大镜像。`proxychains4 curl` 成功也不代表 `docker pull` 会走同一代理，因为
pull 的网络请求由 Docker daemon 发出；不要在多人共用节点自行重启 Docker daemon。

如果 node1 已安装 `skopeo`，可以尝试让用户态进程走现有 proxychains，绕过 daemon
的 registry 出网：

```bash
command -v skopeo
proxychains4 -q skopeo copy --retry-times 5 \
  docker://quay.io/ascend/vllm-ascend:v0.22.1rc1 \
  docker-archive:/data/disk2/glm52-study/vllm-ascend-v0.22.1rc1.tar:quay.io/ascend/vllm-ascend:v0.22.1rc1
docker load --input /data/disk2/glm52-study/vllm-ascend-v0.22.1rc1.tar
bash scripts/07_build_capture_image.sh
```

`command -v skopeo` 无输出时不要自行安装系统包，直接走管理员或离线 tar 路径。

成功时必须出现：

```text
CAPTURE_IMAGE_OK image_ref=glm52-expert-capture:v0.22.1rc1-w8a8-v1 patch_id=glm52-w8a8-logical-topk-v1
```

官方镜像中的 `vllm` 版本可能显示为 `0.22.1+empty`。`+empty` 是 PEP 440
local-version 构建标记，release 仍是 `0.22.1`；构建、启动和 readiness 门禁均接受
`0.22.1` 或 `0.22.1+...`，但仍拒绝 `0.22.2`、`0.22.1rc1` 等其他 release。
该镜像还可能把 `vllm-ascend` 作为 editable source tree 安装在
`/vllm-workspace/`，而不是复制到 `site-packages`。补丁和构建后检查通过 Python
实际 import path 定位源码，不能用 distribution metadata 拼接安装路径。

随后仍在 node1 导出派生镜像：

```bash
source configs/cluster.env
source configs/node.env

export CAPTURE_IMAGE=glm52-expert-capture:v0.22.1rc1-w8a8-v1
export TRANSFER_DIR="${LOCAL_STATE_ROOT}/capture-image"
mkdir -p "${TRANSFER_DIR}"

docker save --output "${TRANSFER_DIR}/glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar" \
  "${CAPTURE_IMAGE}"
sha256sum "${TRANSFER_DIR}/glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar" \
  > "${TRANSFER_DIR}/glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar.sha256"
```

把这两个文件通过已可用的节点内传输方式送到 node0 的
`/data/node0_disk2/glm52-study/capture-image/`。如果 node1 可以直接 SSH 到 node0，
可使用：

```bash
ssh root@7.150.8.22 'mkdir -p /data/node0_disk2/glm52-study/capture-image'
scp "${TRANSFER_DIR}/glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar" \
  "${TRANSFER_DIR}/glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar.sha256" \
  root@7.150.8.22:/data/node0_disk2/glm52-study/capture-image/
```

在 **node0** 导入并核对：

```bash
cd /data/node0_disk2/glm52-study/capture-image
sha256sum -c glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar.sha256
docker load --input glm52-expert-capture-v0.22.1rc1-w8a8-v1.tar
docker image inspect glm52-expert-capture:v0.22.1rc1-w8a8-v1 \
  --format '{{index .Config.Labels "glm52.capture_patch_id"}}'
```

最后在两节点各自的 `configs/cluster.env` 执行下列命令。不要复制 `node.env`，只有
`cluster.env` 应保持一致：

```bash
sed -i \
  -e 's|^RUN_PROFILE=.*|RUN_PROFILE=expert_capture|' \
  -e 's|^IMAGE_REF=.*|IMAGE_REF=glm52-expert-capture:v0.22.1rc1-w8a8-v1|' \
  -e 's|^VLLM_VERSION_OVERRIDE=.*|VLLM_VERSION_OVERRIDE=0.22.1|' \
  -e 's|^ENABLE_ROUTE_CAPTURE=.*|ENABLE_ROUTE_CAPTURE=1|' \
  -e 's|^CAPTURE_PATCH_ID=.*|CAPTURE_PATCH_ID=glm52-w8a8-logical-topk-v1|' \
  -e 's|^EXPECTED_VLLM_PACKAGE_VERSION=.*|EXPECTED_VLLM_PACKAGE_VERSION=0.22.1|' \
  -e 's|^EXPECTED_VLLM_ASCEND_PACKAGE_VERSION=.*|EXPECTED_VLLM_ASCEND_PACKAGE_VERSION=0.22.1rc1|' \
  -e 's|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS=1|' \
  -e 's|^API_BIND_HOST=.*|API_BIND_HOST=127.0.0.1|' \
  configs/cluster.env

grep -E '^(RUN_PROFILE|IMAGE_REF|VLLM_VERSION_OVERRIDE|ENABLE_ROUTE_CAPTURE|CAPTURE_PATCH_ID|EXPECTED_VLLM_PACKAGE_VERSION|EXPECTED_VLLM_ASCEND_PACKAGE_VERSION|MAX_NUM_SEQS|API_BIND_HOST)=' \
  configs/cluster.env
```

修改 `cluster.env` 后，两节点重新执行 00、01、02；随后每次新 run 仍执行 05、06。
`10_launch_node.sh` 会自动带上 `--enable-return-routed-experts`、关闭 async
scheduling 和 prefix cache。模型与数据路径不需要重新下载或复制。

### 10.2 仅在 node0 下载 routing workload

benchmark 数据只写入远端 `DATA_ROOT`。以下命令下载 MMLU-Pro（通用推理）、
SWE-bench Lite（软件工程）、LiveCodeBench（编程）以及确定性的 RULER-style
NIAH 长上下文 workload。它们用于路由分布采集，不替代各 benchmark 的官方评分
harness。

```bash
python3 -m venv .client-venv
.client-venv/bin/pip install -r requirements-client.txt

export DATA_ROOT="${RUN_HOST_ROOT}/benchmark-data"
bash scripts/20_prepare_benchmarks.sh \
  --data-root "${DATA_ROOT}" \
  --benchmarks mmlu_pro,swebench_lite,livecodebench,ruler_niah \
  --limit 50 \
  --ruler-words 2048
```

若下载节点需要代理，只对上述下载命令使用服务器已配置的网络方式。不要让
loopback API 采集经由 HTTP 代理；后续脚本会显式直连 `127.0.0.1`。每个 Hugging
Face 数据集在下载时解析并记录不可变 revision SHA 到
`${DATA_ROOT}/manifests/`。脚本直接加载 MMLU-Pro/SWE-bench Lite 的 Parquet 和
LiveCodeBench 的 JSONL 文件，绝不执行仓库中的 dataset script，因此不会触发
`Dataset scripts are no longer supported`。

`--limit 50` 只限制每个 workload 写出的 JSONL 请求数。MMLU-Pro 和 SWE-bench Lite
仍会缓存选定 split 的 Parquet 文件；LiveCodeBench 在 limit 大于 0 时使用 streaming，
读够指定行数立即停止，不再先下载并解析六个总计数 GiB 的 JSONL。`--limit 0` 取远端
dataset 的全部 rows；`ruler_niah` 没有远端全量集，`--limit 0` 时固定生成 50 条。
`--ruler-words 2048` 是每条合成 NIAH prompt 的词数，不会下载 RULER 数据。

若日志已经显示 SWE-bench Lite 成功，而 LiveCodeBench 因 `datetime is not JSON
serializable` 失败，更新源码后仅重跑最后两个 workload；这不会覆盖已有 MMLU-Pro
和 SWE-bench Lite：

```bash
bash scripts/20_prepare_benchmarks.sh \
  --data-root "${DATA_ROOT}" \
  --benchmarks livecodebench,ruler_niah \
  --limit 50 \
  --ruler-words 2048 \
  --overwrite
```

如果更换为 streaming 后在 `us.aws.cdn.hf.co` 报
`CERTIFICATE_VERIFY_FAILED: self signed certificate in certificate chain`，但之前的日志
已经出现 `Generating test split: 1055 examples`，说明完整的非 streaming Arrow
cache 已经生成。直接复用它，不再访问 CDN：

```bash
bash scripts/20_prepare_benchmarks.sh \
  --data-root "${DATA_ROOT}" \
  --benchmarks livecodebench,ruler_niah \
  --limit 50 \
  --ruler-words 2048 \
  --no-livecodebench-streaming \
  --overwrite
```

这个恢复命令成立的前提是没有删除 `${DATA_ROOT}/hf-cache`。如果缓存不存在，
non-streaming loader 会重新下载完整 split；此时应向管理员取得服务器代理的 CA PEM，
再对下载命令设置 `REQUESTS_CA_BUNDLE=/实际路径/ca.pem` 和
`SSL_CERT_FILE=/实际路径/ca.pem`。不要通过关闭 TLS 校验绕过证书错误。

### 10.3 新 run 启动、采集和分析

按步骤 5–8 用新的 `RUN_ID` 启动 `expert_capture` 服务，并在 node0 看到
`SERVICE_READY` 后执行：

```bash
export DATA_ROOT="${RUN_HOST_ROOT}/benchmark-data"

bash scripts/22_run_benchmark_suite.sh \
  configs/cluster.env configs/node.env \
  --data-root "${DATA_ROOT}" \
  --benchmarks mmlu_pro,swebench_lite,livecodebench,ruler_niah \
  --max-requests 50 \
  --max-tokens 16
```

该脚本先运行 `12_smoke_request.py --require-routes`，再顺序发送请求。每条请求
都保存原始请求、原始响应、逐 token `routes.npy`、prompt/completion token 数和
route SHA-256；中断后使用相同参数加 `--resume` 即可继续。输出在：

```text
${RUN_HOST_ROOT}/${RUN_ID}/node0/benchmarks/
  captures/<benchmark>/routes/*.npy
  captures/<benchmark>/aggregate-counts.npz
  analysis/per-layer-metrics.csv
  analysis/expert-rankings.csv
  analysis/workload-summary.csv
  analysis/hot-set-overlap.csv
  analysis/analysis-summary.json
```

主结论使用 token-expert assignment，不把“一个 token 命中任一热门 expert”当作
主统计。`per-layer-metrics.csv` 中 `top51_assignment_share` 是每层严格 20% 的
51 个 logical experts 承担的 assignment 比例；`k90 <= 51` 才支持该层的
“20% experts 覆盖 90% assignments”命题。`hot-set-overlap.csv` 用每层 top-51
logical expert 的 Jaccard 分数比较不同 workload 是否可共用 HBM 热 expert 集。

## 11. 下次直接运行的最短路径

模型、镜像和 benchmark input 已经在远端时，不需要再次下载。同步更新后的源码到
两节点后：

1. 两节点运行 00、01、02 的 `--confirm-existing-image`；node0 只在 model ready
   marker 丢失或模型变更时运行 03、04。
2. 保持 `expert_capture` 配置不变，在 node0 创建新的
   `expert-capture-<UTC timestamp>` RUN_ID，node1 读取同一值。
3. 两节点运行 05；node0 先、node1 紧接着运行 06；node0 先、node1 紧接着运行 10；
   最后仅 node0 运行 11 并等待 `SERVICE_READY`。
4. node0 运行 22。不要并发多个请求，也不要将请求发往 `NODE0_COORDINATOR_IP`。
5. 结束时两节点分别运行 19 的 `--remove`，保留 run 目录供后续统计和复查。
