# 双节点 Ascend A2 从零部署 GLM-5.2 W8A8

本文给出从“服务器上没有项目源码，但模型权重已经下载完成”开始，到双节点
HCCL 验证、启动 GLM-5.2 和完成最小请求的完整步骤。所有模型、镜像、benchmark
数据和运行结果只存在远端服务器；Mac 只生成几十 KB 的源码包。

当前阶段使用 `vendor_smoke`，目标是先证明 2 节点 × 8 张 Ascend NPU 可以稳定
加载并调用模型。它还不是 expert-load 实验结果；正式 benchmark 必须在 route
capture 门禁通过后再运行。

## 0. 拓扑和不可混淆的边界

| 项目角色 | SSH 管理入口 | 存储标签 | 本机源码建议路径 |
| --- | --- | --- | --- |
| `node0` / `NODE_RANK=0` | `7.150.8.22` | `node0` | `/data/node0_disk2/glm52-study/GLM-VLLM-ASCEND` |
| `node1` / `NODE_RANK=1` | `7.150.15.14` | `node3` | `/data/disk2/glm52-study/GLM-VLLM-ASCEND` |

注意：本项目脚本把第二台计算节点称为 `node1`，但存储清单把它称为
`node3`。`7.150.12.45` 是存储拓扑里的物理 `node1`，不是本项目第二台
计算节点。

以下规则贯穿全部步骤：

- 设备是 Ascend NPU，只使用 `npu-smi`、`hccn_tool`、HCCL、`torch-npu`
  和 `/dev/davinci*`；不要使用 NVIDIA、CUDA 或 NCCL 命令。
- `7.150.8.22` 和 `7.150.15.14` 是 SSH 管理入口，不自动等于
  `LOCAL_IP`、`PEER_IP` 或 `NODE0_COORDINATOR_IP`。
- 服务器禁止保留 `.git`，只部署由 `source_manifest.py bundle` 生成的源码包。
- 模型已经存在，不运行 ModelScope 下载；node0 用 `--adopt-existing` 只读校验。
- 服务器占卡/保活由操作者人工处理。本项目不启动、停止或恢复任何维护脚本。
- 不要停止、删除或复用其他人的容器。确认 0–7 卡确实分配给本次任务后再启动。

本文后续把源码目录记为：

```bash
# node0
PROJECT_ROOT=/data/node0_disk2/glm52-study/GLM-VLLM-ASCEND

# node1
PROJECT_ROOT=/data/disk2/glm52-study/GLM-VLLM-ASCEND
```

每次打开新终端，都先进入：

```bash
cd "${PROJECT_ROOT}/qinyingqi/expert_load"
```

## 1. 在 Mac 生成无 Git 源码包

只在 Mac 的完整 Git 工作树执行：

```bash
cd /Users/qyqsmacbookpro/Desktop/GLM-VLLM-ASCEND/qinyingqi/expert_load

python3 scripts/source_manifest.py generate
python3 scripts/source_manifest.py verify
MANIFEST_SHA256="$(python3 scripts/source_manifest.py digest)"

python3 scripts/source_manifest.py bundle \
  --expected-sha256 "${MANIFEST_SHA256}" \
  --output /tmp/glm52-expert-load-source.tar.gz

BUNDLE_SHA256="$(shasum -a 256 /tmp/glm52-expert-load-source.tar.gz | awk '{print $1}')"
printf 'MANIFEST_SHA256=%s\nBUNDLE_SHA256=%s\n' \
  "${MANIFEST_SHA256}" "${BUNDLE_SHA256}"
```

把这两个值分开保存：

- `MANIFEST_SHA256`：填入两节点 `configs/cluster.env` 的
  `SOURCE_MANIFEST_SHA256`。
- `BUNDLE_SHA256`：仅用于校验传到服务器的 `.tar.gz`。

二者通常不同。把 `BUNDLE_SHA256` 填进 `SOURCE_MANIFEST_SHA256` 会出现
`source manifest SHA-256 mismatch`。

源码包不包含 `.git`、真实配置、模型、benchmark 数据和运行日志。通过团队允许
的传输方式把同一个 `/tmp/glm52-expert-load-source.tar.gz` 传到两台机器；不要在
Mac 下载模型或 benchmark。

## 2. 两节点核对并解压同一个源码包

在每个节点先核对传输包。把 `<MAC记录的BUNDLE_SHA256>` 替换为步骤 1 的值：

```bash
sha256sum /path/to/glm52-expert-load-source.tar.gz
test "$(sha256sum /path/to/glm52-expert-load-source.tar.gz | awk '{print $1}')" = \
  '<MAC记录的BUNDLE_SHA256>'
```

只有 `test` 返回 0 才解压：

```bash
mkdir -p "${PROJECT_ROOT}"
tar -xzf /path/to/glm52-expert-load-source.tar.gz -C "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}/qinyingqi/expert_load"

test ! -e "${PROJECT_ROOT}/.git"
python3 scripts/source_manifest.py verify
```

最后一条必须出现 `SOURCE_MANIFEST_OK`。服务器没有 `.git` 是预期状态；不要在
服务器执行 `source_manifest.py generate`。

## 3. 先做只读环境检查

两节点分别执行：

```bash
uname -m
hostname
npu-smi info
ls -l /dev/davinci{0,1,2,3,4,5,6,7}
ls -l /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
docker version
docker info --format 'DockerRootDir={{.DockerRootDir}}'
df -hT / /data
```

预期 `uname -m` 是 `aarch64`，`npu-smi info` 能看到 8 张本次获准使用的 A2，
并且 Docker 客户端能连接 daemon。

如果出现 `/var/run/docker.sock: permission denied`，应使用管理员分配的 Docker
用户/组或授权方式重新登录。不要用 `chmod 777 /var/run/docker.sock` 绕过权限。

## 4. 找到通信网卡和主机通信 IP

在两节点分别执行并保存输出：

```bash
ip -br -4 addr
ip -4 route
```

从两台机器共同可达的高速通信网段中选择地址，然后交叉验证：

```bash
# 在 node0，用候选 node1 通信 IP 替换占位符
ip -4 route get <NODE1通信IP>

# 在 node1，用候选 node0 通信 IP 替换占位符
ip -4 route get <NODE0通信IP>
```

输出中的 `src` 是本机 `LOCAL_IP`，`dev` 是本机 `LOCAL_NIC`。最终关系必须是：

```text
NODE0_COORDINATOR_IP = node0 LOCAL_IP
node0 PEER_IP        = node1 LOCAL_IP
node1 PEER_IP        = node0 LOCAL_IP = NODE0_COORDINATOR_IP
```

不要只因为 SSH 使用 `7.150.*` 就直接把管理地址写入配置。若候选地址、路由或
防火墙不明确，先向管理员确认承载 HCCL/Gloo 的主机通信网卡。

## 5. 获取 8 个 HCCN NPU 地址

两节点分别执行：

```bash
HCCN_TOOL="$(command -v hccn_tool || printf '%s' /usr/local/Ascend/driver/tools/hccn_tool)"
for device_id in 0 1 2 3 4 5 6 7; do
  printf '\n=== device %s ===\n' "${device_id}"
  "${HCCN_TOOL}" -i "${device_id}" -ip -g
  "${HCCN_TOOL}" -i "${device_id}" -link -g
  "${HCCN_TOOL}" -i "${device_id}" -net_health -g
done
```

按设备编号 0–7 记录每台机器的 8 个 NPU IP：

- node0 的 `configs/remote_npu_ips.txt` 填 node1 的 8 个 NPU IP；
- node1 的 `configs/remote_npu_ips.txt` 填 node0 的 8 个 NPU IP；
- 每行一个地址，严格保持设备 0、1、…、7 的顺序，不写主机 IP。

示例结构：

```text
<对端device0的NPU-IP>
<对端device1的NPU-IP>
<对端device2的NPU-IP>
<对端device3的NPU-IP>
<对端device4的NPU-IP>
<对端device5的NPU-IP>
<对端device6的NPU-IP>
<对端device7的NPU-IP>
```

## 6. 创建两节点配置

### 6.1 两节点完全相同的 `cluster.env`

两节点分别执行：

```bash
cp configs/cluster.env.example configs/cluster.env
```

把下列占位符替换为步骤 1 和步骤 4 的真实值；其余值保持一致：

```bash
CLUSTER_NAME=glm52-a2-2x8
NODE0_COORDINATOR_IP=<node0通信IP>
SOURCE_MANIFEST_SHA256=<步骤1的MANIFEST_SHA256>

MODEL_ID=Eco-Tech/GLM-5.2-w8a8
MODEL_REVISION=edd93687ef1c3417d0b92e2cd01cf67e9e9c0039
MODEL_CONTAINER_PATH=/models/GLM-5.2-w8a8
MODELSCOPE_BIN=/data/node0_disk2/glm52-study/tools/modelscope-venv/bin/modelscope
MODEL_DOWNLOAD_WORKERS=8

RUN_PROFILE=vendor_smoke
IMAGE_REF=quay.io/ascend/vllm-ascend:glm5.2
VLLM_VERSION_OVERRIDE=0.21.0
ENABLE_ROUTE_CAPTURE=0

EXPECTED_VLLM_PACKAGE_VERSION=0.22.1
EXPECTED_VLLM_ASCEND_PACKAGE_VERSION=0.22.1rc1
CAPTURE_PATCH_ID=none

SERVED_MODEL_NAME=glm-52
API_PORT=7000
DP_RPC_PORT=13389
HCCL_TEST_PORT=29501
HCCL_TEST_TIMEOUT_SECONDS=900
NUM_NODES=2
NPUS_PER_NODE=8
TP_SIZE=8
DP_SIZE=2
DP_SIZE_LOCAL=1
MAX_MODEL_LEN=40000
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=16
GPU_MEMORY_UTILIZATION=0.95
BLOCK_SIZE=128
SEED=1024

HEALTH_TIMEOUT_SECONDS=7200
STOP_TIMEOUT_SECONDS=180
MIN_DOCKER_FREE_GIB=80
MIN_MODEL_STORAGE_FREE_GIB=900
```

虽然模型不再下载，`MODELSCOPE_BIN` 和 `MIN_MODEL_STORAGE_FREE_GIB` 保留为完整
集群配置；`--adopt-existing` 不会调用 ModelScope。

### 6.2 node0 的 `node.env`

在 node0：

```bash
cp configs/node0.env.example configs/node.env
```

配置内容：

```bash
NODE_RANK=0
LOCAL_IP=<node0通信IP>
PEER_IP=<node1通信IP>
LOCAL_NIC=<node0通信网卡名>
AUTHORIZED_NPU_IDS=0,1,2,3,4,5,6,7

MODEL_HOST_PATH=/data/node0_disk2/glm52-study/models/GLM-5.2-w8a8
RUN_HOST_ROOT=/data/node0_disk2/glm52-study/runs
LOCAL_STATE_ROOT=/data/node0_disk2/glm52-study/local-state
```

### 6.3 node1 的 `node.env`

在 node1：

```bash
cp configs/node1.env.example configs/node.env
```

配置内容：

```bash
NODE_RANK=1
LOCAL_IP=<node1通信IP>
PEER_IP=<node0通信IP>
LOCAL_NIC=<node1通信网卡名>
AUTHORIZED_NPU_IDS=0,1,2,3,4,5,6,7

MODEL_HOST_PATH=/data/node0_disk2/glm52-study/models/GLM-5.2-w8a8
RUN_HOST_ROOT=/data/node0_disk2/glm52-study/runs
LOCAL_STATE_ROOT=/data/disk2/glm52-study/local-state
```

`MODEL_HOST_PATH` 和 `RUN_HOST_ROOT` 在两节点必须是同一个可见绝对路径。
node1 通过挂载看到 node0 的 `/data/node0_disk2`。只有 `LOCAL_STATE_ROOT` 使用
本机数据盘，因此 node1 写 `/data/disk2/...`。

### 6.4 清理 CRLF 并加载

两节点执行：

```bash
sed -i 's/\r$//' \
  configs/cluster.env configs/node.env configs/remote_npu_ips.txt
file configs/cluster.env configs/node.env configs/remote_npu_ips.txt

source configs/cluster.env
source configs/node.env
printf 'rank=%s local=%s peer=%s nic=%s model=%s\n' \
  "${NODE_RANK}" "${LOCAL_IP}" "${PEER_IP}" "${LOCAL_NIC}" "${MODEL_HOST_PATH}"
```

`file` 不应再显示 `with CRLF line terminators`。否则 `source` 可能出现
`: command not found` 或变量末尾带 `\r`。

## 7. 核对共享存储和现有模型

在 node0：

```bash
source configs/cluster.env
source configs/node.env
mkdir -p "${RUN_HOST_ROOT}/operator"
printf 'node0-storage-ok\n' > "${RUN_HOST_ROOT}/operator/storage-probe.txt"
test -r "${MODEL_HOST_PATH}/config.json"
find "${MODEL_HOST_PATH}" -maxdepth 1 -type f -name '*.safetensors' | wc -l
```

在 node1：

```bash
source configs/cluster.env
source configs/node.env
cat "${RUN_HOST_ROOT}/operator/storage-probe.txt"
test -r "${MODEL_HOST_PATH}/config.json"
find "${MODEL_HOST_PATH}" -maxdepth 1 -type f -name '*.safetensors' | wc -l
```

node1 必须读到 `node0-storage-ok`。若模型实际放在另一个目录，先通过服务器
允许的方式把它放到两节点都能以同一绝对路径读取的位置，或者同步修改两节点
`MODEL_HOST_PATH`；不要在 Mac 中转 774 GB 权重。

## 8. 校验源码身份；SHA mismatch 的正确处理

两节点执行：

```bash
source configs/cluster.env
python3 scripts/source_manifest.py verify \
  --expected-sha256 "${SOURCE_MANIFEST_SHA256}"
```

成功标志是 `SOURCE_MANIFEST_OK`。如果失败：

```bash
python3 scripts/source_manifest.py verify
ACTUAL_MANIFEST_SHA256="$(python3 scripts/source_manifest.py digest)"
printf 'configured=%s\nactual=%s\n' \
  "${SOURCE_MANIFEST_SHA256}" "${ACTUAL_MANIFEST_SHA256}"
```

- 不带 expected 的验证成功：文件与包内清单一致。只有当步骤 2 的压缩包哈希已
  与 Mac 的可信 `BUNDLE_SHA256` 核对过，才把两节点配置改为 Mac 记录的
  `MANIFEST_SHA256`。最常见错误是错填了 `.tar.gz` 的哈希。
- 不带 expected 的验证也失败：源码文件被修改、漏传或混合了两个版本。重新
  解压同一个已核对 `BUNDLE_SHA256` 的源码包；不要在服务器重新生成清单。

两节点还应核对 `cluster.env` 字节完全一致：

```bash
sha256sum configs/cluster.env
```

把两边输出进行比较；必须相同。

## 9. 两节点预检和 HCCN 点对点连通

配置最终确定后，两节点分别执行：

```bash
bash scripts/00_preflight.sh \
  configs/cluster.env configs/node.env

bash scripts/01_hccn_ping.sh \
  configs/cluster.env configs/node.env configs/remote_npu_ips.txt
```

必须分别看到：

```text
PREFLIGHT_OK
HCCN_PING_OK
```

脚本会在共享 `RUN_HOST_ROOT/gates/node0` 和 `node1` 写门禁。之后只要修改
`cluster.env`、`node.env` 或源码清单，就必须在对应节点重跑 00 和 01。

## 10. 准备两节点完全相同的 Ascend 镜像

### 10.1 两节点都能访问 registry

分别执行：

```bash
bash scripts/02_pull_image.sh \
  configs/cluster.env configs/node.env --confirm-pull
```

### 10.2 只有 node1 能拉取时

在 node1 拉取并导出到数据盘：

```bash
source configs/cluster.env
docker pull "${IMAGE_REF}"
docker save --output /data/disk2/glm52-study/vllm-ascend-glm52.tar "${IMAGE_REF}"
sha256sum /data/disk2/glm52-study/vllm-ascend-glm52.tar \
  > /data/disk2/glm52-study/vllm-ascend-glm52.tar.sha256
```

通过节点间允许的传输方式把 tar 和 sha256 文件送到 node0 数据盘。在 node0：

```bash
cd /data/node0_disk2/glm52-study
sha256sum -c vllm-ascend-glm52.tar.sha256
docker load --input vllm-ascend-glm52.tar
```

若 `docker load` 报 `permission denied`，先解决 Docker daemon 用户权限；tar
文件本身可读不代表当前用户有权访问 daemon。

镜像已经存在后，两节点都执行：

```bash
bash scripts/02_pull_image.sh \
  configs/cluster.env configs/node.env --confirm-existing-image
docker image inspect "${IMAGE_REF}" --format '{{.Id}}'
```

两节点镜像 ID 必须完全相同，且两边都出现 `PULL_OK`。如果
`proxychains4 curl https://quay.io/v2/` 返回 HTTP 401，通常说明网络已经到达
Quay 的 Registry API；匿名探测收到 401 不等于镜像一定需要登录。`docker pull`
超时还可能是 Docker daemon 没有使用 shell 中的代理。

## 11. 接管并严格校验已经下载好的模型

只在 node0 执行：

```bash
source configs/cluster.env
source configs/node.env

bash scripts/03_download_model.sh \
  configs/cluster.env configs/node.env --adopt-existing

bash scripts/04_model_manifest.sh \
  configs/cluster.env configs/node.env
```

预期依次出现：

```text
ADOPTED_OK
MANIFEST_OK
```

`--adopt-existing` 不联网、不下载、不改模型权重；它先运行严格校验，再为当前
`MODEL_ID`、`MODEL_REVISION` 和路径创建状态记录。快速模型门禁应确认 182 个
indexed safetensors，所有 tensor payload 共 `773854402152` 字节，实际 shard
文件共 `773876016944` 字节且 `problems=[]`。

node1 不执行 03 或 04，只确认共享 ready 文件可见：

```bash
source configs/cluster.env
source configs/node.env
test -r "$(dirname "${MODEL_HOST_PATH}")/.model-ready/GLM-5.2-w8a8.ready"
```

## 12. 创建两节点共享的 RUN_ID

只在 node0 生成一次：

```bash
source configs/cluster.env
source configs/node.env
mkdir -p "${RUN_HOST_ROOT}/operator"
export RUN_ID="vendor-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "${RUN_ID}" > "${RUN_HOST_ROOT}/operator/current-run-id"
printf 'RUN_ID=%s\n' "${RUN_ID}"
```

在 node1 读取：

```bash
source configs/cluster.env
source configs/node.env
export RUN_ID="$(tr -d '[:space:]' < "${RUN_HOST_ROOT}/operator/current-run-id")"
printf 'RUN_ID=%s\n' "${RUN_ID}"
```

两边输出必须相同。打开新终端后需要重新 `source` 两个配置并重新 `export
RUN_ID=...`。失败重试使用新的 `RUN_ID`，不要覆盖已经启动过的轮次。

## 13. 人工确认 NPU 空闲并写本轮 NPU 门禁

先与同事/调度者确认当前 0–7 卡属于本次任务，再在两节点检查：

```bash
npu-smi info
docker ps --no-trunc
```

一个 Docker daemon 可以同时运行多个容器，NPU 容器也不限制只能有一个；但
两个容器不能在未协调的情况下争用同一组 NPU。看到其他人的 NPU 容器时不要
停止或删除，先确认设备归属。

确认后，两节点分别在已经导出相同 `RUN_ID` 的终端执行：

```bash
bash scripts/05_prepare_npus.sh \
  configs/cluster.env configs/node.env
```

成功标志是 `NPU_READY`。此脚本只记录状态；不会调用 keep-alive 或 stop 脚本。

## 14. 运行 16-rank HCCL 集合通信门禁

准备两个终端。先在 node0 执行，随后立即在 node1 执行同一命令：

```bash
bash scripts/06_hccl_collective.sh \
  configs/cluster.env configs/node.env
```

两边都必须出现 `HCCL_COLLECTIVE_OK`。任何一边失败，都先保存两边日志并停止
仍在等待的对端命令；修复网络/端口/镜像差异后用新的 `RUN_ID` 重试。

## 15. 启动 GLM-5.2 双节点服务

先在 node0：

```bash
bash scripts/10_launch_node.sh \
  configs/cluster.env configs/node.env
```

看到 `LAUNCH_OK` 后，立即在 node1 执行相同命令：

```bash
bash scripts/10_launch_node.sh \
  configs/cluster.env configs/node.env
```

两边都出现 `LAUNCH_OK` 后，在 node0 等待 API：

```bash
bash scripts/11_wait_ready.sh \
  configs/cluster.env configs/node.env
```

模型很大，首次加载可能持续较久。成功标志是 `SERVICE_READY`。观察日志时只读取
本轮容器：

```bash
CONTAINER_NAME="glm52-${RUN_PROFILE}-node${NODE_RANK}-${RUN_ID}"
docker logs --tail 200 --timestamps "${CONTAINER_NAME}"
```

## 16. 发送最小推理请求

只需在 node0 建一次轻量客户端环境；这里不会安装模型框架或下载 benchmark：

```bash
python3 -m venv .client-venv
.client-venv/bin/pip install -r requirements-client.txt

.client-venv/bin/python scripts/12_smoke_request.py \
  --base-url "http://${NODE0_COORDINATOR_IP}:${API_PORT}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --output-dir "${RUN_HOST_ROOT}/${RUN_ID}/client-smoke"
```

请求成功只证明模型服务能生成文本。`vendor_smoke` 没有 route capture，不能把
其输出用于判断“20% expert 是否处理 90% tokens”。

## 17. 停止本轮服务

实验完成后，在 node0、node1 分别执行：

```bash
bash scripts/19_stop_node.sh \
  configs/cluster.env configs/node.env --remove
```

脚本只处理当前 `RUN_ID`、当前节点和本项目 ownership label 对应的容器，不会
操作其他人的容器。随后按服务器管理员要求人工恢复占卡状态；本项目不代办。

## 18. 关键成功标志和证据位置

| 阶段 | 终端成功标志 | 主要证据位置 |
| --- | --- | --- |
| 源码 | `SOURCE_MANIFEST_OK` | `SOURCE_MANIFEST.json` |
| 预检 | `PREFLIGHT_OK` | `${RUN_HOST_ROOT}/preflight/` |
| HCCN ping | `HCCN_PING_OK` | `${RUN_HOST_ROOT}/connectivity/` |
| 镜像 | `PULL_OK` | `${RUN_HOST_ROOT}/image-manifests/` 和 `gates/node*/image.gate` |
| 现有模型接管 | `ADOPTED_OK` | `${RUN_HOST_ROOT}/model-adopt/` |
| 模型校验 | `MANIFEST_OK` | `${RUN_HOST_ROOT}/model-manifests/` |
| NPU 状态 | `NPU_READY` | `${RUN_HOST_ROOT}/${RUN_ID}/node*/npu-ready/` |
| HCCL collective | `HCCL_COLLECTIVE_OK` | `${RUN_HOST_ROOT}/${RUN_ID}/node*/hccl-collective/` |
| 容器启动 | `LAUNCH_OK` | `${RUN_HOST_ROOT}/${RUN_ID}/node*/` |
| API | `SERVICE_READY` | node0 的 `SERVICE_READY`、health 和 models 响应 |
| 请求 | 客户端退出码 0 | `${RUN_HOST_ROOT}/${RUN_ID}/client-smoke/` |

## 19. 常见错误速查

| 错误 | 含义 | 处理 |
| --- | --- | --- |
| `: command not found`，`file` 显示 CRLF | 配置是 Windows 换行 | `sed -i 's/\r$//' configs/*.env` 后重新 `source` |
| Docker socket `permission denied` | 当前用户无 daemon 权限 | 使用管理员分配的 Docker 用户/组；不要 chmod socket |
| Quay `context deadline exceeded` | Docker daemon 到 registry 网络不通/未走代理 | 配 daemon 代理，或在可拉取节点 `docker save` 后传输 |
| `curl .../v2/` 返回 401 | 已到 registry API，但匿名请求被 challenge | 不等同于网络失败；继续检查实际 `docker pull` 和 daemon 代理 |
| `source manifest SHA-256 mismatch` | 常见为错把 tarball SHA 填入配置 | 先无 expected 验证，再比较 trusted manifest SHA；无需动模型 |
| `source file SHA-256 mismatch for ...` | 源码文件变动/混包 | 重新核对 tarball SHA 并解压干净包，不在远端 generate |
| `missing preflight/hccn gate` | 当前配置尚未跑门禁，或配置改过 | 两节点按顺序重跑 00、01 |
| image gate 的 ID 不同 | 两节点镜像内容不同 | 传输同一镜像 tar，重新执行 02 |
| `revision-bound download state is missing` | 现有模型尚未接管 | node0 运行 `03_download_model.sh ... --adopt-existing` |
| 模型 `problems=[]` 但旧脚本 total mismatch | 旧校验把辅助 `rot.safetensors` 算法混淆 | 使用当前 `validate_model_files.py`，核对三组精确字节数 |
| HCCN/HCCL 失败 | NPU IP 顺序、通信 NIC、端口或链路错误 | 核对 device 0–7 对端 IP、`route get`、link/net_health 和端口 |
| 发现其他 NPU 容器 | daemon 上存在其他容器，不代表只能有一个 | 不操作他人容器；确认卡归属和显存占用后再继续 |
| API 长时间未 ready | 模型仍加载、容器退出、HCCL 或内存错误 | 查看两节点本轮容器日志和 `${RUN_ID}` 证据，不盲目重启 |

## 20. 从 vendor smoke 进入 expert benchmark 的门禁

当前代码只能完成可靠的服务基线，还不能生成可用于论文/汇报的 expert-load
分布。正式实验至少还需完成：

1. 构建与锁定 GLM-5.2 W8A8 route-capture 派生镜像；
2. 服务启用 `--enable-return-routed-experts`，并用
   `12_smoke_request.py --require-routes` 验证返回真实、非空的 logical expert ID；
3. 在远端固定版本下载 benchmark，至少覆盖通用知识/推理、代码任务、长上下文、
   agent tool-use 和 SWE/OpenHands 类任务；
4. 为每个样本保存 benchmark、split、sample ID、request ID、prompt/completion
   token、逐 token/逐层 top-8 expert ID、并发度、seed 和运行版本；
5. 分 benchmark 统计 expert assignment share、top-20% share、达到 90% 所需最小
   expert 数，并报告层间与 prefill/decode 差异；
6. route gate 未通过前，不批量下载和运行 benchmark，避免只得到文本结果却无法
   回答 expert 负载问题。

也就是说，本手册跑通后得到的是后续实验的可信运行底座，而不是提前宣称
“20% expert 处理 90% token”的结论。
