# Atlas A2 上的 GLM-5 / GLM-5.2 部署基线

核对日期：2026-07-22（Asia/Shanghai）

## 1. 结论

以官方 GLM 模型页中的实际示例为准，第一阶段应锁定：

| 项目 | 基线 |
| --- | --- |
| vLLM | `0.22.1` |
| vLLM Ascend | `0.22.1rc1` |
| A2 镜像 | `quay.io/ascend/vllm-ascend:v0.22.1rc1` |
| Python | `>=3.10, <3.13` |
| CANN | 标签 Dockerfile 和兼容矩阵为 `9.0.0` |
| PyTorch / torch-npu | 标签源码均锁定 `2.10.0` |
| Triton Ascend | `3.2.1` |
| Mooncake | `0.3.9` |
| Transformers | 标签源码锁定 `5.5.4`；GLM-5 文档要求至少 `5.2.0` |

源码和提交号见项目根目录的 `UPSTREAM.lock`。

## 2. A2 资源要求

### 部署验证边界

官方示例给出的是最低资源与命令模板，不代表任意 A2 环境已经具备运行条件。实际部署前必须逐节点确认服务器型号、NPU 数量与显存、驱动、固件、CANN、操作系统、容器运行时和 HCCL 连通性；多节点方案还必须确认所有节点的软件栈与网络拓扑一致。

- `GLM-5-w4a8` 的官方示例需要单台 8 卡 A2；
- GLM-5.2 的量化方案至少需要 2 台同构 A2，BF16 方案需要更多节点；
- “满足卡数”不能替代权重完整性、量化格式、通信和真实推理请求验证。

### GLM-5

官方单机 A2 示例仅明确覆盖量化模型 `GLM-5-w4a8`：

| 模型 | A2 资源 | 官方示例拓扑 | 示例上下文 |
| --- | --- | --- | --- |
| `GLM-5-w4a8` | 1 台 Atlas 800 A2，8 × 64 GiB | `DP1 TP8`，开启 EP | `max-model-len=32768` |

示例还设置 `max-num-seqs=2`、Ascend 量化、Chunked Prefill、Prefix Caching 和 3 个 MTP speculative tokens。完整命令可从官方页面查看；对应主线文档的精确提交号记录在 `UPSTREAM.lock`。

### GLM-5.2

官方模型权重章节给出的 A2 最低节点数为：

| 模型 | A2 节点需求 |
| --- | --- |
| `GLM-5.2` BF16 | 4 台 Atlas 800 A2，每台 8 × 64 GiB |
| `GLM-5.2-w8a8` | 2 台 Atlas 800 A2，每台 8 × 64 GiB |
| `GLM-5.2-w4a8c8` | 2 台 Atlas 800 A2，每台 8 × 64 GiB |

当前 A2 在线部署示例重点覆盖 `GLM-5.2-w4a8c8` 的双节点方案。每个节点的命令使用本地 `DP1 × TP8`，全局 `DP2`，开启 EP；示例配置 `max-model-len=40000`、`max-num-seqs=16` 和 5 个 MTP speculative tokens。

## 3. 必须先处理的上游差异

### 3.1 latest 文档与发布标签不是同一份内容

用户给出的 URL 指向 `latest`，其源码来自 vLLM Ascend `main`。本地 `v0.22.1rc1` 标签中也有 GLM 文档，但 GLM-5.2 的模型变体和部署模板与当前网页不同。因此项目使用同一个 vLLM Ascend 子模块定位两个版本，不复制第二套文档目录：

- 子模块工作树：可构建的 `v0.22.1rc1` 标签源码及标签内文档；
- 精确提交 `44fc51ffb18dd05ca53c6509eae0058ba3c39333`：用户所给 `latest` 页面对应的主线文档。

离线查看主线版本可执行：

```bash
git -C upstream/vllm-ascend fetch origin 44fc51ffb18dd05ca53c6509eae0058ba3c39333
git -C upstream/vllm-ascend show 44fc51ffb18dd05ca53c6509eae0058ba3c39333:docs/source/tutorials/models/GLM5.md
git -C upstream/vllm-ascend show 44fc51ffb18dd05ca53c6509eae0058ba3c39333:docs/source/tutorials/models/GLM5.2.md
```

后续修改任何命令时，必须注明依据的是“标签基线”还是“latest 文档模板”。

### 3.2 CANN 9.0.0 与 9.0.1 的差异

- `v0.22.1rc1` 兼容矩阵和该标签的 A2 `Dockerfile` 都使用 CANN `9.0.0`；
- 当前 main 安装页模板已经写成 CANN `9.0.1`，这反映了主线开发环境，而不是已锁定标签的容器基线。

首轮复现优先使用官方 `v0.22.1rc1` A2 镜像，避免手工混配。若必须源码安装，应先根据服务器现有驱动/CANN 决定是严格复现标签，还是整体升级到更新的 vLLM Ascend 配对，不能只单独替换 CANN。

### 3.3 GLM-5.2 A2 命令中的 `VLLM_VERSION=0.21.0`

latest GLM-5.2 A2 双节点命令显式设置了 `VLLM_VERSION=0.21.0`，但安装页和兼容矩阵将 `vllm-ascend 0.22.1rc1` 配对到 `vllm 0.22.1`。该环境变量会影响 vLLM Ascend 的兼容代码路径。

首轮实验应原样记录并验证该变量的实际作用，不应在没有 A2 运行证据前擅自删除，也不应据此把整个源码仓降级成 vLLM 0.21.0。

### 3.4 GLM-5.2 A2 卡数描述冲突

同一份 latest GLM-5.2 文档中：

- 权重章节写的是 2 台 A2、每台 `64G × 8`；
- A2 部署标题写成 2 台 A2、`64G × 32`；
- Docker 和服务命令每台只暴露/使用 8 张卡，拓扑也是每节点 `DP1 × TP8`。

因此 `64G × 32` 不能直接作为采购或拓扑依据。开工前应以真实服务器的 `npu-smi info`、节点数量和 HCCL 拓扑为准。

### 3.5 更晚的 v0.23.0rc1 已出现，但模型页未切换

2026-07-22 的 latest 兼容矩阵已列出 `vllm-ascend 0.23.0rc1` / `vllm 0.23.0`，但 GLM-5 和 GLM-5.2 模型页仍指定 `v0.22.1rc1` 镜像。为了先获得可对照的官方复现结果，本项目暂时使用模型页锁定版本；升级应单独做成后续对照实验。

## 4. 开工前检查清单

1. 记录每台服务器型号、A2 卡数和每卡显存。
2. 记录驱动、固件、CANN、NNAL、OS、Python 版本。
3. 双节点及以上先完成 HCCL 多机通信验证，再启动模型。
4. 确认节点间使用的 NIC、IP、HCCL/GLOO/TP 网卡名和端口开放情况。
5. 选择模型变体和权重来源，并校验文件完整性；不要混用不同量化格式。
6. 首轮优先使用官方 A2 镜像和未经改动的官方命令，保存容器 digest、环境变量、完整日志和请求结果。
7. 得到基线结果后，再在项目 `main` 上提交经过验证且已脱敏的配置、脚本和结果摘要。
8. 遵守部署环境的 NPU 调度和保活策略；任务结束后恢复其改变的运行状态，并记录实际使用的设备集合。

本次只准备文档和源码，没有下载模型权重、容器镜像，也没有在 A2 服务器上执行验证。
