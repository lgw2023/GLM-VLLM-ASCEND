# GLM-5.2 expert-load on Ascend A2

本目录用于在两台 Atlas 800 A2（每台 8 张 Ascend NPU）上启动
`Eco-Tech/GLM-5.2-w8a8`，验证双节点 HCCL，并为后续 expert 路由采集和
benchmark 实验提供运行入口。

所有远端命令都使用 Ascend 原生组件：`npu-smi`、`hccn_tool`、HCCL、
`torch-npu` 和 `/dev/davinci*`。不要替换成 CUDA、NCCL 或 NVIDIA 命令。

## 运行原则

- 服务器上的 NPU 占用由操作者手工管理，本项目不会调用服务器维护脚本。
- 执行步骤 05 前，操作者自行用 `npu-smi info` 确认 0–7 卡可用。
- 旧调试轮次产生的本地状态目录不再参与启动，也不需要删除。
- 模型、镜像和 benchmark 数据只放远端服务器，不下载到 Mac。
- 两台节点必须使用相同 `RUN_ID`、相同镜像 ID 和相同模型路径。

## 1. 发布无 Git 源码包

服务器禁止保存 `.git`。在 Mac 的完整 Git 工作树中执行：

```bash
cd qinyingqi/expert_load
python3 scripts/source_manifest.py generate
python3 scripts/source_manifest.py verify
python3 scripts/source_manifest.py bundle \
  --output /tmp/glm52-expert-load-source.tar.gz

sha256sum SOURCE_MANIFEST.json
sha256sum /tmp/glm52-expert-load-source.tar.gz
```

把压缩包传到两台服务器并从 `GLM-VLLM-ASCEND` 根目录解压。不要传模型、
benchmark 数据、运行日志或真实配置文件。

将 `SOURCE_MANIFEST.json` 的 SHA-256 填到两节点相同的
`configs/cluster.env`：

```bash
SOURCE_MANIFEST_SHA256=<64位小写SHA-256>
```

远端验证：

```bash
source configs/cluster.env
python3 scripts/source_manifest.py verify \
  --expected-sha256 "${SOURCE_MANIFEST_SHA256}"
```

## 2. 配置两节点

```bash
cp configs/cluster.env.example configs/cluster.env
cp configs/node0.env.example configs/node.env  # node0
cp configs/node1.env.example configs/node.env  # node1
cp configs/remote_npu_ips.txt.example configs/remote_npu_ips.txt
```

两节点 `cluster.env` 必须一致。`NODE0_COORDINATOR_IP`、`LOCAL_IP`、
`PEER_IP` 和 `LOCAL_NIC` 必须使用承载 HCCL/Gloo 的通信网络地址，不能直接
假设 SSH 管理地址可用。

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
bash scripts/04_model_manifest.sh configs/cluster.env configs/node.env
```

node1 使用相同的共享模型路径，不重新下载模型。

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
  --base-url "http://${NODE0_COORDINATOR_IP}:${API_PORT}/v1" \
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

## 10. Expert capture 和 benchmark 当前门禁

正式 expert-load 实验必须让服务返回真实 `routed_experts`。当前同步包还没有
GLM-5.2 W8A8 route-capture 派生镜像和 benchmark 适配器，因此不能把空值、
零数组或文本输出当作 expert 分布。

下一步顺序是：

1. 核对远端镜像内 vLLM 和 vLLM-Ascend 版本；
2. 启用 `--enable-return-routed-experts` 并通过
   `scripts/12_smoke_request.py --require-routes`；
3. 固定并下载 MMLU-Pro、LiveCodeBench、RULER、tau2-bench 和 SWE/OpenHands
   小样本到远端；
4. 分 benchmark 保存逐 token、逐层、top-8 logical expert ID；
5. 统计 top-51 expert assignment share 和达到 90% 所需的最小 expert 数。

在 route gate 通过前，不启动大规模 benchmark，避免得到无法分析的文本结果。
