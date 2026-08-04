# GLM-5.1：Node3 单机 8 卡 A2 最短部署与 1 条 LiveCodeBench 路由测试

这个目录只做一件事：在 Node3（`7.150.15.14`）的 8 张 Ascend 910B1 上，先把
GLM-5.1 服务跑通，再用已经准备好的 LiveCodeBench 第 1 条数据验证 expert 路由。

不会修改 `qinyingqi/deepseek_expert_load` 或 `qinyingqi/expert_load`。模型下载、镜像构建、
NPU 启动均只应在远端 Node3 执行，不能在 Mac 上执行。

## 0. 量化选择

本实验固定使用 `Eco-Tech/GLM-5.1-w4a8`：

- Node3 是 Atlas 800 A2，8 × 64 GiB HBM；W8A8 不作为单机 8 卡方案。
- vLLM-Ascend 的 GLM-5/5.1 共用教程中，A2 单机 TP8 示例明确使用 GLM5-W4A8；
  W8A8 的 A2 测试配置是双节点。因此这里把同架构的 GLM-5.1 W4A8 作为最快候选，
  由模型审计和实际启动决定是否通过，不把它写成未经验证的兼容性结论。
- GLM-5.1 配置应为 `glm_moe_dsa`、78 层、前 3 层 dense、其后 75 层 MoE、256 个
  routed experts、每 token top-8。脚本从模型 `config.json` 读取并验证，不写死到结果里。

首轮为了正确性，明确关闭 MTP、图模式、prefix cache、chunked prefill、FlashComm、
动态 EPLB 和 balance scheduling；并使用 `MAX_NUM_SEQS=1`。这不是吞吐配置。
启动脚本保留 `VLLM_ASCEND_MLA_PARALLEL=1`，因为官方 release note 对 W4A8 eager
模式给出了该 OOM 规避项。

官方参考：

- <https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html>
- <https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html>
- <https://docs.vllm.ai/projects/ascend/en/latest/user_guide/release_notes.html>

## 1. Node3 拉代码并创建配置

```bash
cd /home/qinyingqi/GLM-VLLM-ASCEND/qinyingqi/glm51_expert_load

cp configs/node3.env.example configs/node3.env
sed -i 's/\r$//' configs/node3.env
```

默认模型路径是：

```text
/data/disk1/Public/GLM-5.1-w4a8
```

先在 Node3 找一下是否已经存在模型：

```bash
find /data/disk1 /data/disk2 /data/disk3 /data/node0_disk1 \
  -maxdepth 4 -type d -iname '*GLM*5.1*w4a8*' 2>/dev/null
```

如果找到其他路径，只改 `configs/node3.env` 中的 `MODEL_HOST_PATH`。

### 模型不存在时才下载（仍然只在 Node3）

下面使用魔乐社区官方 `openmind_hub` 下载，不创建 `.git` 目录：

```bash
python3 -m pip install --user -U openmind_hub

export GLM51_MODEL_DIR=/data/disk1/Public/GLM-5.1-w4a8
export HUB_WHITE_LIST_PATHS="${GLM51_MODEL_DIR}"

python3 - "${GLM51_MODEL_DIR}" <<'PY'
from openmind_hub import snapshot_download
import sys

snapshot_download(
    repo_id="Eco-Tech/GLM-5.1-w4a8",
    local_dir=sys.argv[1],
    local_dir_use_symlink=False,
)
print("GLM51_DOWNLOAD_OK", sys.argv[1])
PY
```

不要在这里下载 benchmark。配置默认直接复用已经存在的：

```text
/data/node0_disk2/glm52-study/runs/benchmark-data/inputs/livecodebench.jsonl
```

## 2. 只读模型审计

这一步不加载 tensor，只检查模型类型、W4A8 标记、分片是否非空和总字节数：

```bash
bash scripts/00_audit_model.sh configs/node3.env
```

成功必须看到：

```text
"compatible": true
```

若显示 W8A8，说明模型目录选错；不要继续启动。若分片超过 480 GiB，脚本只给出
显眼警告而不额外阻塞；真正的是否能装入 HBM 由启动过程决定。

## 3. 构建 route-capture 镜像（不占 NPU）

先看 Node3 是否已经有基础镜像：

```bash
docker image inspect quay.io/ascend/vllm-ascend:v0.22.1rc1 \
  --format '{{.Id}}'
```

有镜像时直接构建，不访问 quay.io：

```bash
bash scripts/01_build_capture_image.sh configs/node3.env
```

只有基础镜像确实不存在、且 Node3 能访问 quay.io 时才执行：

```bash
bash scripts/01_build_capture_image.sh \
  configs/node3.env --confirm-pull-base
```

成功标志：

```text
CAPTURE_IMAGE_OK image=glm51-expert-capture:v0.22.1rc1-w4a8-v2
```

这个派生镜像只在 MoE 通信切分之前记录逻辑 expert ID，并补齐 TP=8 下的 route
all-gather；不改模型权重。

## 4. 确认 8 张卡空闲，然后启动

先人工查看；不要停止别人的进程或容器：

```bash
npu-smi info
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'
```

管理员确认 0-7 都分配给你后，手动停止这 8 张卡上的 keep-alive，再启动：

```bash
bash scripts/02_launch_tp8_w4a8.sh \
  configs/node3.env \
  --confirm-npu-ids 0,1,2,3,4,5,6,7
```

成功看到 `LAUNCH_OK` 后，模型仍可能需要较长时间加载。另开一个终端可查看：

```bash
bash scripts/05_status.sh configs/node3.env
```

## 5. 等服务就绪

```bash
bash scripts/03_wait_ready.sh configs/node3.env
```

必须等到：

```text
EXPERT_PLACEMENT_OK ep_size=8 experts_per_rank=32 unique_experts=256 duplicates=0
SERVICE_READY url=http://127.0.0.1:7200/v1 model=glm-5.1-w4a8
```

`127.0.0.1` 是刻意设置的，避免本地健康检查走服务器代理或外部访问策略。
`${RUN_ROOT}/${RUN_ID}/expert-placement.json` 会保存 8 个 rank 的完整 expert 列表；
验证要求每个 rank 恰好 32 个、跨 rank 无重复、0-255 无缺失、动态 EPLB 关闭且冗余
expert 数为 0。只有该验证通过后才会生成 `service.ready`。

## 6. 跑 LiveCodeBench 第 1 条并验证 expert 路由

```bash
bash scripts/04_run_livecodebench_one.sh configs/node3.env
```

成功标志：

```text
LIVECODEBENCH_ONE_OK
```

查看结果：

```bash
source configs/node3.env
RUN_ID="$(tr -d '[:space:]' < "${RUN_ROOT}/current-run-id")"
RESULT_DIR="${RUN_ROOT}/${RUN_ID}/livecodebench-one"

python3 -m json.tool "${RESULT_DIR}/summary.json"
ls -lh "${RESULT_DIR}"
```

关键产物：

- `summary.json`：shape、token 数、覆盖 expert 数、全局 top-20 experts。
- `routes.raw.npy`：完整路由，期望 shape 为
  `(prompt_tokens + completion_tokens - 1, 78, 8)`。
- `aggregate-counts.npz`：`total/prefill/decode × 75 MoE layers × 256 experts` 计数。
- `request.json`、`response.json`：不含超大的 base64 route 字段，便于检查。

脚本会严格拒绝以下情况：route shape 不一致、dense 层出现 route、expert ID 越界、
top-8 重复、route 恒定/陈旧。失败时 `routes.raw.npy` 和 `response.json` 仍会保留。

## 7. 停止并移除本次容器

```bash
bash scripts/06_stop.sh configs/node3.env --remove
```

然后按管理员约定，手动恢复同一组 0-7 卡的 keep-alive。脚本不会触碰其他容器。

## 最短命令清单

模型和 benchmark 已经存在、基础镜像也已经存在时，只需：

```bash
cd /home/qinyingqi/GLM-VLLM-ASCEND/qinyingqi/glm51_expert_load
cp configs/node3.env.example configs/node3.env
sed -i 's/\r$//' configs/node3.env

bash scripts/00_audit_model.sh configs/node3.env
bash scripts/01_build_capture_image.sh configs/node3.env
bash scripts/02_launch_tp8_w4a8.sh configs/node3.env \
  --confirm-npu-ids 0,1,2,3,4,5,6,7
bash scripts/03_wait_ready.sh configs/node3.env
bash scripts/04_run_livecodebench_one.sh configs/node3.env
bash scripts/06_stop.sh configs/node3.env --remove
```
