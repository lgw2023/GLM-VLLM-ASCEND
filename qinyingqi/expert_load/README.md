# GLM-5.2 Expert Load on Ascend A2

这是同步到两台 Atlas A2 节点的第一阶段运行包。当前目标是闭合环境、镜像、模型、NPU keep-alive、双节点 HCCL 和 GLM-5.2 vendor smoke；暂不下载完整 benchmark，也不把未打补丁的 W8A8 路由数据当作实验结果。

Mac 上的可信 Git 工作树只用于 review、生成 source manifest、提交和本地单测。远端服务器不保留 `.git`：部署包使用逐文件 SHA-256 manifest 和共享配置中的固定 manifest digest 建立源码身份。下面第 2–10 节命令都应在远端 Ascend 服务器执行；部署源码不会自动下载模型、拉镜像、停止 NPU 或启动服务。模型下载还必须显式传入 `--confirm-large-download`。

固定基线：

- 2 nodes × 8 Ascend 910B1（64 GiB HBM/卡）；
- `Eco-Tech/GLM-5.2-w8a8`；
- global DP2、local DP1、TP8、EP enabled；
- 组内 vLLM `0.22.1` 与 vLLM-Ascend `0.22.1rc1`；
- vendor smoke 镜像 `quay.io/ascend/vllm-ascend:glm5.2`；
- node0 提供 API，node1 只运行 headless DP worker。

所有脚本仅使用 Ascend 原生组件：`npu-smi`、`hccn_tool`、HCCL、CANN、`torch-npu` 和 `/dev/davinci*`。不要替换成 CUDA、NCCL 或 NVIDIA 命令。

## 目录内容

```text
configs/      两节点配置模板；真实 IP/NIC 配置不会进入 Git
scripts/      预检、HCCN/HCCL、模型、启动、请求、停服与恢复脚本
tests/        route response 和模型分片校验单测
SOURCE_MANIFEST.json  不依赖 .git 的受管源码逐文件 SHA-256 清单
patches/      下一阶段 W8A8 route-capture 补丁位置
benchmarks/   下一阶段 benchmark 固定版本与适配器位置
BASELINE.md   实验口径和 20%/90% 判定定义
MODEL_PROVENANCE.md  固定模型 revision、元数据 hash 和字节口径
```

共享证据写到 `RUN_HOST_ROOT`；NPU 租约、恢复状态和已校验的 keep-alive 脚本副本写到各节点本地的 `LOCAL_STATE_ROOT`。两者都不会写入仓库。脚本不会登录另一台服务器；下面标注“两节点”的步骤需要你在两个 SSH 终端分别执行。

## 1. 生成无 Git 发布包并配置两节点

先在 Mac 的完整、可信 Git 工作树完成代码修改。确认两个上游 submodule 位于 `BASELINE.md` 固定的 commit 且没有本地修改后，在本目录生成 manifest、验证并构建不含 `.git` 的确定性发布包：

```bash
cd /Users/qyqsmacbookpro/Desktop/GLM-VLLM-ASCEND/qinyingqi/expert_load

python3 scripts/source_manifest.py generate
python3 scripts/source_manifest.py verify
python3 scripts/source_manifest.py bundle \
  --output /tmp/glm52-expert-load-source.tar.gz
```

`generate` 只允许在本地 Git 工作树运行，并核对两个 submodule 的固定 commit；`verify` 和后续远端脚本完全不需要 Git。记录输出中的 `source_id=sha256:<64-hex>`，把不含 `sha256:` 前缀的 64 位值填入两节点 `configs/cluster.env` 的 `SOURCE_MANIFEST_SHA256`。任何受管文件变化后都必须重新生成 manifest、重新分发整个发布包并重跑门禁。

先 review、commit 并 push 这些源码和 `SOURCE_MANIFEST.json`。再通过获批的传输通道把 `/tmp/glm52-expert-load-source.tar.gz` 传到两个节点。推荐每个 source ID 解压到一个新的发布目录，避免旧文件与新版本混用：

```bash
mkdir -p <REMOTE_RELEASE_ROOT>
tar -xzf glm52-expert-load-source.tar.gz -C <REMOTE_RELEASE_ROOT>
cd <REMOTE_RELEASE_ROOT>/qinyingqi/expert_load

python3 scripts/source_manifest.py verify \
  --expected-sha256 <SOURCE_MANIFEST_SHA256>
```

发布包只包含本运行包的受管源码和 manifest，不包含 `.git`、模型、benchmark、运行结果或真实节点配置。首次部署时执行：

```bash
umask 077
cp configs/cluster.env.example configs/cluster.env
```

升级已有部署时，从旧发布目录复制 `configs/cluster.env`、本节点的 `configs/node.env` 和 `configs/remote_npu_ips.txt`，然后把新的 `SOURCE_MANIFEST_SHA256` 同步写入两节点 `cluster.env`。不要把 node0 的 `node.env` 复制给 node1。

node0 复制：

```bash
cp configs/node0.env.example configs/node.env
```

node1 复制：

```bash
cp configs/node1.env.example configs/node.env
```

编辑两份本地配置。`NODE0_COORDINATOR_IP`、`LOCAL_IP`、`PEER_IP` 必须是实际承载 HCCL/Gloo 的地址，不能直接拿 SSH 管理地址代替。两节点分别确认：

```bash
source configs/cluster.env
source configs/node.env
ip route get "${PEER_IP}"
ip -br addr show dev "${LOCAL_NIC}"
npu-smi info
sha256sum "${KEEPALIVE_STOP_SCRIPT}" "${KEEPALIVE_START_SCRIPT}"
```

把两个小写 SHA-256 填入 `node.env`。确认当前 0–7 卡健康、无他人任务且本次获准使用后，才把 `NPU_USE_CONFIRMED=NO` 改成 `YES`。这些配置文件在 `.gitignore` 中，不要强制加入 Git。

`cluster.env` 必须在两节点逐字节相同，包括预先填写的数据盘 `MODELSCOPE_BIN` 绝对路径；即使该可执行文件只在 node0 使用，也不要在通过门禁后单边修改配置。用 `sha256sum configs/cluster.env` 比较两边结果。

node1 模板默认通过 `/data/node0_disk2` 使用 node0 导出的共享模型和运行目录；如果服务器上的实际挂载不同，必须以 `findmnt -T PATH` 和 `df -hT PATH` 的现场结果修改，不能假设两台机器的本地绝对路径相同。

## 2. 两节点预检

两节点分别执行：

```bash
bash scripts/00_preflight.sh configs/cluster.env configs/node.env
```

它检查 aarch64、8 个 Davinci 设备、NPU 状态、驱动挂载源、NIC/路由、端口、每卡 HCCN link/health、磁盘、Docker、完整 source manifest 和 keep-alive 脚本 identity。源码或配置修改后旧门禁自动失效，必须重跑。

## 3. 两节点 HCCN 逐卡网络检查

这一步是 HCCN L3 ping，不等价于 HCCL 集合通信。先在每台机器查看本机 8 张卡的 NPU IP：

```bash
for card in {0..7}; do
  hccn_tool -i "${card}" -ip -g
done
```

在每台机器创建对端卡 0–7 的 IP 清单：

```bash
cp configs/remote_npu_ips.txt.example configs/remote_npu_ips.txt
```

填好后，两节点分别执行：

```bash
bash scripts/01_hccn_ping.sh \
  configs/cluster.env configs/node.env configs/remote_npu_ips.txt
```

## 4. 两节点拉取并核对镜像

先看 `docker info` 的 `DockerRootDir` 和 `df -hT`。现有环境记录显示 node0 空间余量较紧；默认门禁要求 pull 完成后仍至少保留 80 GiB。如果预算不足，先由管理员确认精确清理目标或迁移 DockerRoot，不能用 `docker system prune` 做全局清理。

两节点分别执行：

```bash
bash scripts/02_pull_image.sh \
  configs/cluster.env configs/node.env --confirm-pull
```

脚本在 pull 前后检查 DockerRootDir 空间，并记录 image ID、RepoDigest、镜像内实际 package version 和 capture-patch label。两节点 image ID 必须一致；后续启动会强制消费这两个共享门禁。

如果镜像已通过 `docker save`、传输文件 SHA-256、`docker load` 的流程从另一节点导入，不需要再次访问 quay.io；核对本地 `IMAGE_REF` 后用下面的只验证模式重建当前 source ID 对应的 image gate：

```bash
bash scripts/02_pull_image.sh \
  configs/cluster.env configs/node.env --confirm-existing-image
```

## 5. node0 下载并验证模型

模型约 774 GB，只在共享模型目录下载一次。先把 ModelScope CLI 放到数据盘 venv，不要安装到紧张的根分区：

```bash
python3 -m venv /data/node0_disk2/glm52-study/tools/modelscope-venv
/data/node0_disk2/glm52-study/tools/modelscope-venv/bin/pip install -U modelscope
```

模板已经把 `MODELSCOPE_BIN` 指向上述数据盘路径；创建 venv 不需要再修改已经过门禁的 `cluster.env`。然后只在 node0 执行：

```bash
bash scripts/03_download_model.sh \
  configs/cluster.env configs/node.env --confirm-large-download
bash scripts/04_model_manifest.sh \
  configs/cluster.env configs/node.env
```

manifest 会固定校验官方 metadata hash、78 层、前 3 层 dense、256 个 routed experts、top-8、W8A8/W8A8_DYNAMIC 描述、safetensors header，以及 index 引用的 182 个文件、tensor payload 总量和实际文件总量。需要记录完整权重 SHA-256 清单时才追加 `--full-sha256`；它会顺序读取约 774 GB，不要与服务并行。

模板已固定到 [ModelScope `Eco-Tech/GLM-5.2-w8a8`](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8/) 的 40 位 commit；不要改回 `master`/`main` 或其他可移动分支。细节见 `MODEL_PROVENANCE.md`。

## 6. 创建唯一 RUN_ID 并准备 NPU

只在 node0 生成一次 run ID，并写到两节点可见的共享目录：

```bash
source configs/cluster.env
source configs/node.env
mkdir -p "${RUN_HOST_ROOT}/operator"
export RUN_ID="vendor-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "${RUN_ID}" >"${RUN_HOST_ROOT}/operator/current-run-id"
```

node1 读取完全相同的字符串，不能独立调用第二次 `date`：

```bash
source configs/cluster.env
source configs/node.env
export RUN_ID="$(tr -d '[:space:]' <"${RUN_HOST_ROOT}/operator/current-run-id")"
```

两节点再次执行 `npu-smi info`，确认本轮仍可使用，然后在各自终端设置一次性确认并停止官方 keep-alive：

```bash
export NPU_LAUNCH_CONFIRMATION="${RUN_ID}"
bash scripts/05_prepare_npus.sh \
  configs/cluster.env configs/node.env --confirm-stop-keepalive
```

脚本只操作明确授权的 0–7 卡，获取节点级原子租约，保存停用前/后的 marker 与 PGID identity，并在准备失败时自动恢复同一组卡。prepare、HCCL、launch、stop、restore 还由本地 lifecycle 原子锁串行化。恢复所需脚本和状态保存在本机数据盘，即使 node0 的共享挂载中断，node1 仍有恢复材料。

## 7. 同时运行真实 16-rank HCCL 门禁

先在 node0 启动下面命令；它会等待 node1。立即在 node1 的另一个终端执行同一命令：

```bash
bash scripts/06_hccl_collective.sh configs/cluster.env configs/node.env
```

该步骤在两节点各启动 8 个 `torch-npu` rank，实际执行 HCCL `all_reduce` 和 `all_to_all`。只有两边都出现 `HCCL_COLLECTIVE_OK` 才能启动模型。任一侧失败会恢复该侧 keep-alive；此时还要在另一侧执行恢复命令，并为下一次尝试创建新的 `RUN_ID`：

```bash
bash scripts/20_restore_npus.sh configs/cluster.env configs/node.env
```

## 8. 启动双节点 vendor smoke

保持两个终端中的 `RUN_ID` 和 `NPU_LAUNCH_CONFIRMATION` 不变。先 node0、随后立即 node1 分别执行：

```bash
bash scripts/10_launch_node.sh configs/cluster.env configs/node.env
```

node0 绑定 `${LOCAL_IP}:${API_PORT}`；node1 使用 `--headless --data-parallel-start-rank 1`，不会错误传入正数 `--api-server-count`。两边的 `--data-parallel-address` 都指向 node0。

启动脚本要求预检、HCCN、两节点同一镜像、模型 ready、两节点 HCCL 和本轮 keep-alive 状态全部通过。若本节点启动失败，会停止本脚本创建的容器并恢复本节点 keep-alive；如果另一节点已经启动，还需在那里运行停服脚本。

启动成功后脚本不会常驻充当跨节点 watchdog。实验期间应在两个终端持续查看容器状态；任一节点容器后续异常退出时，立即停止 peer 节点的本轮容器，再在两节点分别运行第 10 节停服/恢复命令。不要在另一 NPU 容器仍运行时手动启动 keep-alive。

在 node0 等待 API：

```bash
bash scripts/11_wait_ready.sh configs/cluster.env configs/node.env
```

## 9. node0 发出 smoke 请求

客户端只额外依赖 NumPy：

```bash
python3 -m venv .client-venv
.client-venv/bin/pip install -r requirements-client.txt

.client-venv/bin/python scripts/12_smoke_request.py \
  --base-url "http://${NODE0_COORDINATOR_IP}:${API_PORT}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --output-dir "${RUN_HOST_ROOT}/${RUN_ID}/client-smoke"
```

vendor smoke 只验证真实权重能够加载、双节点能够生成，不要求 `routed_experts`。这一步成功后再进入 W8A8 capture patch 和 benchmark 阶段。

## 10. 两节点安全停服并恢复 NPU

停止发送请求后，在 node0、node1 分别执行：

```bash
bash scripts/19_stop_node.sh configs/cluster.env configs/node.env
```

脚本校验保存的 container ID 和四个 ownership labels，只停止当前 run 的精确容器；随后恢复同一组 0–7 卡，并记录 `stopped_card_ids`、`restored_card_ids`、`restoration_status`。默认保留停止后的容器用于排障；确认日志后可再次执行：

```bash
bash scripts/19_stop_node.sh configs/cluster.env configs/node.env --remove
```

如果模型容器从未成功创建但 keep-alive 仍处于 `PREPARED`，直接运行：

```bash
bash scripts/20_restore_npus.sh configs/cluster.env configs/node.env
```

如果进程被 `kill -9` 或主机异常中断，lifecycle 锁会故意保留并 fail-closed。先查看 `${LOCAL_STATE_ROOT}/runs/${RUN_ID}/lifecycle.lock/owner.env`、确认其中 PID 已不存在，再检查 `docker ps --no-trunc`、`npu-smi info` 和 keep-alive state；只有确认没有仍在执行的操作或 NPU consumer 后，才把整个锁目录 `mv` 到同一数据盘的 `.stale.<timestamp>` 名称，再重试恢复。不要直接删除不明锁文件。

节点级 NPU lease `${LOCAL_STATE_ROOT}/npu-leases/cards-0-7` 同样包含 `owner.env`。如果 run state 已存在，保留 lease 并优先运行 `20_restore_npus.sh`；只有在 owner PID 已消失、该 run 尚无 state、无 NPU consumer 且 keep-alive 已验证为原始 running 状态时，才可把整个 lease 目录原子移动为 `.stale.<timestamp>`。不满足这些条件就停止操作并请管理员复核。

不要使用 `killall`、`pkill -f`、`docker system prune`，也不要运行会全局清理其他用户任务的 performance 脚本。

## 正式 expert capture 的硬门禁

当前仓库还没有 W8A8 route-capture patch 和派生镜像，因此不要把 vendor smoke 的空/null route 当成 expert 分布。下一阶段派生镜像必须：

- 安装 vLLM `0.22.1` 与 vLLM-Ascend `0.22.1rc1`；
- 带非 `none` 的 Docker label `glm52.capture_patch_id`；
- 使用不可变 `MODEL_REVISION`；
- 配置 `RUN_PROFILE=expert_capture`、`ENABLE_ROUTE_CAPTURE=1`、`MAX_NUM_SEQS=1`；
- 显式关闭 async scheduling、prefix cache、dynamic EPLB、balance scheduling、fused MC2，并使用 eager mode；
- 通过下面的 route gate：

```bash
.client-venv/bin/python scripts/12_smoke_request.py \
  --base-url "http://${NODE0_COORDINATOR_IP}:${API_PORT}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --output-dir "${RUN_HOST_ROOT}/${RUN_ID}/route-gate" \
  --require-routes --ignore-eos --max-tokens 4
```

门禁会发送四个确定性请求：主请求、完全相同的重复请求、同 prompt 的 `max_tokens=1` 请求，以及不同 prompt 的 contrast 请求。它要求 Base64 `.npy` 为 `uint8`、shape 为 `(P+G-1, 78, 8)`、dense 0–2 层为零、MoE 3–77 层 top-8 唯一；同时要求重复请求的 token IDs 与完整 route tensor 一致、1-token/4-token 的 prefill 边界一致、不同 prompt 的重叠 MoE prefill route 至少有一行变化，从而拒绝位置/层相关但输入无关的陈旧模板。

当前同步包到这里为止。`benchmarks/README.md` 只列出下一阶段候选任务，没有下载 benchmark、没有 route-capture patch，也没有产生任何 expert-load 结果；必须先让派生镜像通过上述 route gate，再固定 benchmark commit 和下载适配器。

## push 前本地验证

```bash
python3 scripts/source_manifest.py generate
python3 scripts/source_manifest.py verify
bash -n scripts/*.sh scripts/lib/*.sh
.client-venv/bin/python -m unittest discover -s tests -v
git -C ../.. status --short --ignored -- qinyingqi/expert_load
python3 scripts/source_manifest.py bundle \
  --output /tmp/glm52-expert-load-source.tar.gz
```

需要提交的是本目录中的 tracked 模板、脚本、文档和 `SOURCE_MANIFEST.json`；不要提交 `configs/cluster.env`、`configs/node.env`、`configs/remote_npu_ips.txt`、模型、运行日志、response 或 `.npy` route 数据。生成 manifest 后不要再修改受管文件；如有修改，重新运行 `generate`。
