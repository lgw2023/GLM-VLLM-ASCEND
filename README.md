# GLM / vLLM Ascend on Atlas A2

本项目用于准备在昇腾 Atlas 800 A2 上运行 GLM-5 与 GLM-5.2，集中保存锁定的上游源码版本、公开部署基线、三人工作目录和共享交流区。

> 当前状态：已完成文档和源码基线准备，尚未下载模型权重、容器镜像或 Python/CANN 依赖，也尚未在 A2 服务器上完成 GLM 运行验证。

## 获取项目

两个源码仓以 Git 子模块固定到经过核对的提交：

```bash
git clone --recurse-submodules https://github.com/lgw2023/GLM-VLLM-ASCEND.git
cd GLM-VLLM-ASCEND
```

已经普通克隆但缺少源码时：

```bash
git submodule update --init --recursive
```

## 从这里开始

| 文件 | 职责 |
| --- | --- |
| [`docs/A2_DEPLOYMENT_BASELINE.md`](docs/A2_DEPLOYMENT_BASELINE.md) | GLM 模型资源要求、软件版本、上游差异与开工检查清单 |
| [`交流区/README.md`](交流区/README.md) | 三人传递需求、交接和共同决策的规则与入口 |
| [`UPSTREAM.lock`](UPSTREAM.lock) | 可机器核对的仓库、提交、镜像和文档快照锁定值 |

## 锁定基线

| 组件 | 版本 | 提交 |
| --- | --- | --- |
| vLLM | `v0.22.1` | `0decac0d96c42b49572498019f0a0e3600f50398` |
| vLLM Ascend | `v0.22.1rc1` | `5f6faa0cb8830f667266f3b8121cd1383606f2a1` |
| A2 Docker 镜像 | `quay.io/ascend/vllm-ascend:v0.22.1rc1` | 官方模型页指定 |
| 用户所给 `latest` 文档 | `main@44fc51ffb18dd05ca53c6509eae0058ba3c39333` | 精确提交号记录在 `UPSTREAM.lock` |

虽然兼容矩阵可能出现更晚版本，本项目第一阶段仍使用 GLM 模型页实际采用的版本组合，避免首次复现前混配。

## 源码布局

| 目录 | 职责 |
| --- | --- |
| `upstream/vllm` | vLLM 核心推理引擎子模块，固定到 `v0.22.1` |
| `upstream/vllm-ascend` | Ascend 平台插件、算子、构建和模型部署文档子模块，固定到 `v0.22.1rc1` |

两者是配套依赖，不是同一代码的两份副本。项目仓只维护一个 `main` 主分支；上游子模块保持在锁定提交，任何升级都应同时更新 `UPSTREAM.lock` 并重新核对兼容性。

## 三人工作目录

| 开发者 | 个人目录 |
| --- | --- |
| liguowei | `liguowei/` |
| qinyingqi | `qinyingqi/` |
| yangyinyue | `yangyinyue/` |

三个目录都是项目根目录下的普通文件夹，不是 Git worktree，也不包含重复源码。个人方案、脚本、实验记录和待合入成果放在本人目录。

共享的 `交流区/` 同样直接位于项目根目录，并在其下划分 `liguowei/`、`qinyingqi/`、`yangyinyue/` 三个独立交流空间。交流区不再嵌入或链接到个人工作目录。

## 官方文档

- [GLM-5 官方页面](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.html#51-single-node-online-deployment)
- [GLM-5.2 官方页面](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html#multi-node-deployment)

发布标签对应的文档位于 `upstream/vllm-ascend/docs/`。用户给出的 `latest` 页面来自更新的主线提交；需要精确离线读取时执行：

```bash
git -C upstream/vllm-ascend fetch origin 44fc51ffb18dd05ca53c6509eae0058ba3c39333
git -C upstream/vllm-ascend show 44fc51ffb18dd05ca53c6509eae0058ba3c39333:docs/source/tutorials/models/GLM5.md
git -C upstream/vllm-ascend show 44fc51ffb18dd05ca53c6509eae0058ba3c39333:docs/source/tutorials/models/GLM5.2.md
```

## 发布边界

公开仓不包含内部服务器地址、认证信息、模型权重、容器镜像、运行日志或私有基础设施清单。服务器侧验证结果必须先脱敏，再进入 `main`。
