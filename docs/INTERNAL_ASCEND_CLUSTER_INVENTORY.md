# 内部昇腾双节点与共享存储清单

> 快照日期：2026-07-22（Asia/Shanghai）
> 信息来源：用户提供的两台服务器 `df -h`、节点 0 的公共模型目录清单与 `docker images` 输出
> 分发级别：内部敏感基础设施信息；只允许保存在权限受控的内部项目中

本文记录用户确认为 `GLM-VLLM-ASCEND` 项目入口的两个逻辑节点、跨机磁盘挂载、公共模型目录和节点 0 镜像清单。它是时点快照，不是实时监控面板；“已知入口”不等于节点 3 已通过 NPU 或模型计算验收，使用前应重新采集容量、挂载和运行状态。

本文件是面向 GLM 项目的同步副本，主事实副本位于 `AK-Infer-Lab/通信模块/docs/internal-ascend-cluster-inventory.md`。节点、容量、模型或镜像清单更新时，两份文件必须同轮更新并保留相同快照日期，避免漂移。

## 1. 登录与命名边界

| 项目逻辑节点 | IP | 登录用户 | 存储拓扑标签 | 本机数据盘路径 |
| --- | --- | --- | --- | --- |
| 节点 0 | `7.150.8.22` | `root` | `node0` | `/data/node0_disk1`、`/data/node0_disk2`、`/data/node0_disk3` |
| 节点 3 | `7.150.15.14` | `root` | `node3` | `/data/disk1`、`/data/disk2`、`/data/disk3`；从其他主机导出时显示为 `/data/node3_disk*` |

安全要求：本文不保存登录口令、私钥、Token 或其他认证材料。登录凭据必须从权限受控的密码管理系统或服务器本地密钥存储获取，不得写入 Git、任务交接文档或命令历史。当前凭据曾通过对话明文提供，应在继续使用前轮换。

需要特别区分两套编号：

- “项目逻辑节点 0/1”是本项目可登录和使用的两台机器编号。
- 挂载路径中的 `node0/node1/node2/node3` 是共享存储拓扑标签。
- 因此，项目逻辑节点 3（`7.150.15.14`）在共享存储拓扑中标记为 `node3`，不是 `node1`。
- `7.150.12.45`（存储标签 `node1`）和 `7.150.14.170`（存储标签 `node2`）目前只确认其磁盘可经网络挂载访问；本文不据此推断它们是项目可登录的计算节点。
- `IP:/data/...` 形式表明这些目录是网络挂载；现有输出没有单独确认具体协议和挂载参数。

SSH 端口、跳板/堡垒机、密钥认证方式和允许登录的来源网络仍需向管理员确认。

## 2. 节点 0 磁盘快照（`7.150.8.22`）

### 2.1 容量重点

- 根分区 `/`：1007G，总使用率 90%，仅余 101G。
- Docker overlay 位于根分区，快照同为 90%；拉取、解包或构建大镜像前必须先检查 Docker 数据目录和根分区余量。
- 三块本地 NVMe 均为 7.0T：`node0_disk1` 使用 73%，`node0_disk2` 使用 4%，`node0_disk3` 使用 10%。
- 节点还能看到存储拓扑中 `node1`、`node2`、`node3` 的三组远端磁盘。

### 2.2 完整 `df -h` 快照

```text
Filesystem                      Size  Used Avail Use% Mounted on
tmpfs                           151G  4.9M  151G   1% /run
/dev/sda2                      1007G  866G  101G  90% /
tmpfs                           754G  8.1G  746G   2% /dev/shm
tmpfs                           5.0M     0  5.0M   0% /run/lock
/dev/nvme0n1p1                  7.0T  5.1T  1.9T  73% /data/node0_disk1
/dev/sda1                       1.1G  8.0M  1.1G   1% /boot/efi
/dev/nvme1n1p1                  7.0T  216G  6.8T   4% /data/node0_disk2
/dev/nvme2n1p1                  7.0T  709G  6.3T  10% /data/node0_disk3
7.150.14.170:/data/node2_disk1  7.0T  2.8T  4.3T  40% /data/node2_disk1
7.150.15.14:/data/node3_disk3   7.0T   52G  7.0T   1% /data/node3_disk3
7.150.12.45:/data/node1_disk1   7.0T  5.9T  1.2T  84% /data/node1_disk1
7.150.14.170:/data/node2_disk2  7.0T   50G  7.0T   1% /data/node2_disk2
7.150.12.45:/data/node1_disk2   7.0T  4.4T  2.7T  63% /data/node1_disk2
7.150.15.14:/data/node3_disk1   7.0T  3.0T  4.1T  43% /data/node3_disk1
7.150.12.45:/data/node1_disk3   7.0T   50G  7.0T   1% /data/node1_disk3
7.150.14.170:/data/node2_disk3  7.0T   50G  7.0T   1% /data/node2_disk3
7.150.15.14:/data/node3_disk2   7.0T   50G  7.0T   1% /data/node3_disk2
tmpfs                           151G   19M  151G   1% /run/user/0
tmpfs                           151G     0  151G   0% /run/user/1002
tmpfs                           151G     0  151G   0% /run/user/1003
tmpfs                           151G     0  151G   0% /run/user/1005
overlay                        1007G  866G  101G  90% /var/lib/docker/overlay2/1d2f76928501c60255c0a7a49948eb1571bc2e53e9f36cf0fca6c8bb65ae4c06/merged
tmpfs                           151G     0  151G   0% /run/user/1001
```

## 3. 节点 3 磁盘快照（`7.150.15.14`）

### 3.1 容量重点

- 根分区 `/`：1007G，使用率 7%，可用 899G。
- 三块本地 NVMe 均为 7.0T，挂载名为 `/data/disk1`、`/data/disk2`、`/data/disk3`；使用率分别为 43%、1%、1%。
- 节点 3 能看到存储拓扑中 `node0`、`node1`、`node2` 的三组远端磁盘。
- 节点 3 的本地 `/data/disk*` 从节点 0 看时，以 `7.150.15.14:/data/node3_disk*` 导出并挂载到 `/data/node3_disk*`。这个本地路径与导出路径差异必须在脚本和容器挂载中显式处理。

### 3.2 完整 `df -h` 快照

```text
Filesystem                      Size  Used Avail Use% Mounted on
tmpfs                           151G  4.6M  151G   1% /run
/dev/sda2                      1007G   67G  899G   7% /
tmpfs                           754G  960K  754G   1% /dev/shm
tmpfs                           5.0M     0  5.0M   0% /run/lock
/dev/nvme0n1p1                  7.0T  3.0T  4.1T  43% /data/disk1
/dev/sda1                       1.1G  8.0M  1.1G   1% /boot/efi
/dev/nvme1n1p1                  7.0T   50G  7.0T   1% /data/disk2
/dev/nvme2n1p1                  7.0T   52G  7.0T   1% /data/disk3
7.150.8.22:/data/node0_disk3    7.0T  709G  6.3T  10% /data/node0_disk3
7.150.12.45:/data/node1_disk2   7.0T  4.4T  2.7T  63% /data/node1_disk2
7.150.14.170:/data/node2_disk3  7.0T   50G  7.0T   1% /data/node2_disk3
7.150.8.22:/data/node0_disk1    7.0T  5.1T  1.9T  73% /data/node0_disk1
7.150.14.170:/data/node2_disk1  7.0T  2.8T  4.3T  40% /data/node2_disk1
7.150.12.45:/data/node1_disk3   7.0T   50G  7.0T   1% /data/node1_disk3
7.150.8.22:/data/node0_disk2    7.0T  216G  6.8T   4% /data/node0_disk2
7.150.14.170:/data/node2_disk2  7.0T   50G  7.0T   1% /data/node2_disk2
7.150.12.45:/data/node1_disk1   7.0T  5.9T  1.2T  84% /data/node1_disk1
tmpfs                           151G   40K  151G   1% /run/user/0
```

## 4. 跨机存储拓扑

| 存储标签 | 存储主机 IP | 三盘路径 | 在项目节点上的可见性 | 当前边界 |
| --- | --- | --- | --- | --- |
| `node0` | `7.150.8.22` | `/data/node0_disk1`、`/data/node0_disk2`、`/data/node0_disk3` | 逻辑节点 0 本地；逻辑节点 3 网络挂载 | 已确认项目入口；节点 0 硬件/运行时另有实测 |
| `node1` | `7.150.12.45` | `/data/node1_disk1`、`/data/node1_disk2`、`/data/node1_disk3` | 两个项目节点均可见为网络挂载 | 只确认存储可见，不确认登录/计算授权 |
| `node2` | `7.150.14.170` | `/data/node2_disk1`、`/data/node2_disk2`、`/data/node2_disk3` | 两个项目节点均可见为网络挂载 | 只确认存储可见，不确认登录/计算授权 |
| `node3` | `7.150.15.14` | 本机为 `/data/disk1`、`/data/disk2`、`/data/disk3`；导出为 `/data/node3_disk1`、`/data/node3_disk2`、`/data/node3_disk3` | 逻辑节点 3 本地；逻辑节点 0 网络挂载 | 已确认项目入口；NPU/运行时待实测 |

容量决策应以“执行命令所在节点看到的路径”为准。跨节点运行容器、Ray、HCCL 或模型服务时，不得假设同一物理磁盘在两台机器上的绝对路径完全相同。

## 5. 公共模型目录快照

目录：`/data/node0_disk1/Public`。下表只证明目录在用户提供的 `ll` 快照中存在；`ll` 显示的 4.0K/8.0K 是目录项大小，不是模型权重实际占用，也不证明权重完整、哈希一致或能被目标运行时加载。

| 目录名 | 所有者 | 用户组 | 列表时间 |
| --- | --- | --- | --- |
| `DeepSeek-V4-Flash` | `root` | `shareddata` | Jul 9 23:55 |
| `DeepSeek-V4-Flash-w8a8-mtp` | `root` | `shareddata` | Jul 17 21:27 |
| `Qwen2.5-3B-Instruct` | `root` | `shareddata` | Jul 10 16:20 |
| `Qwen3.5-0.8B` | `root` | `shareddata` | Jul 6 17:26 |
| `Qwen3.5-27B` | `root` | `shareddata` | Jul 10 19:24 |
| `Qwen3.5-2B` | `root` | `shareddata` | Jul 6 18:04 |
| `Qwen3.5-35B-A3B` | `root` | `shareddata` | Jul 10 20:02 |
| `Qwen3.5-397B-A17B` | `root` | `shareddata` | Jul 10 17:59 |
| `Qwen3.5-4B` | `root` | `shareddata` | Jun 26 11:09 |
| `Qwen3.5-9B` | `root` | `shareddata` | Jul 10 17:44 |
| `Qwen3.6-27B` | `root` | `shareddata` | Jul 3 12:24 |
| `Qwen3.6-27B-FP8` | `root` | `shareddata` | Jul 3 11:56 |
| `Qwen3-Embedding-4B` | `root` | `shareddata` | May 14 12:04 |
| `Qwen3-Embedding-8B` | `nobody` | `shareddata` | May 14 12:13 |
| `Qwen3-Reranker-4B` | `root` | `shareddata` | May 14 11:56 |
| `Qwen3-Reranker-8B` | `nobody` | `shareddata` | May 14 12:26 |

当前清单没有 GLM 权重目录。GLM 项目仍需单独确认模型名称、来源 revision、绝对路径、量化格式、文件清单、总字节数和哈希。

## 6. 节点 0 Docker 镜像快照

| Repository | Tag | Image ID | Created | Size | 已有证据边界 |
| --- | --- | --- | --- | ---: | --- |
| `quay.io/ascend/cann` | `9.0.0-910b-ubuntu22.04-py3.12` | `d8b5c3dbfbf4` | 2 months ago | 11.3GB | 既有记录确认单卡基础容器冒烟通过 |
| `quay.io/ascend/cann` | `9.0.0-910b-ubuntu22.04-py3.11` | `d24afa915a3e` | 2 months ago | 11.3GB | 既有记录确认单卡基础容器冒烟通过 |
| `swr.cn-southwest-2.myhuaweicloud.com/mep-dev-ga/vllm_ascend` | `910B_0.13.0rc0.20260417141425` | `5b4c8865b5f9` | 3 months ago | 30.8GB | 只确认镜像存在 |
| `swr.cn-southwest-2.myhuaweicloud.com/huaweiccs-hivoice-product-ga/mep-vllm-ascend` | `1.0.0.20260320165527` | `98824e745c5c` | 4 months ago | 13.1GB | 只确认镜像存在 |
| `quay.io/ascend/vllm-ascend` | `v0.11.0` | `bd312fe62114` | 7 months ago | 17.7GB | 只确认镜像存在 |
| `swr.cn-southwest-2.myhuaweicloud.com/mep-dev-ga/mep-vllm-ascend` | `11.3.10.300.2` | `1a8dd22f222b` | 8 months ago | 9.92GB | 只确认镜像存在 |
| `swr.cn-southwest-2.myhuaweicloud.com/huaweiccs-hivoice-product-ga/vllm-ascend-0.10.2-910b-cann8.2.rc1-torch2.7.1rc1` | `1.2.9.300` | `dd2d5c1b80c8` | 9 months ago | 20.7GB | 只确认镜像存在 |
| `quay.io/ascend/vllm-ascend` | `v0.10.0rc1` | `ed946500e4cd` | 11 months ago | 15.6GB | 只确认镜像存在 |
| `quay.io/ascend/vllm-ascend` | `v0.9.2rc1` | `e0eb8dc337c1` | 12 months ago | 14.4GB | 只确认镜像存在 |

除两套 CANN 基础镜像已有单卡基础冒烟证据外，其余条目只证明镜像存在于节点 0 的本地镜像库，不代表已通过当前服务器、GLM、模型量化格式或多卡/多节点验收。

本项目当前锁定的目标镜像 `quay.io/ascend/vllm-ascend:v0.22.1rc1` 不在本快照中。由于节点 0 根分区和 Docker overlay 已达 90%，加载该镜像前必须先确认存储落点、镜像大小、回收边界和充足余量。

## 7. 使用前刷新清单

以下命令均为只读采集；不得把凭据或带凭据的远程 URL 写入结果。

两台机器分别执行通用检查：

```bash
hostname
df -h
df -ih
docker images --digests --no-trunc
docker info --format '{{.DockerRootDir}}'
docker system df -v
docker_root_dir=$(docker info --format '{{.DockerRootDir}}')
findmnt -T "${docker_root_dir}"
df -hT "${docker_root_dir}"
df -ih "${docker_root_dir}"
```

节点 0 检查本地盘和公共资产：

```bash
df -hT / /data/node0_disk1 /data/node0_disk2 /data/node0_disk3
findmnt -T /data/node0_disk1
find /data/node0_disk1/Public -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

节点 3 检查本地盘；不要用 `/data/node0_disk*` 代替本机 `/data/disk*`：

```bash
df -hT / /data/disk1 /data/disk2 /data/disk3
findmnt -T /data/disk1
```

双节点 GLM 推理前还必须分别确认两台机器的 NPU 型号、卡数、驱动/CANN、HCCL 网卡与连通性。当前硬件、运行时和 NPU 实测基线仅在 `7.150.8.22` 上闭合；不能仅凭节点 3 的磁盘快照推断它与节点 0 完全同构。
