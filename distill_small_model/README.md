# OpenRouter Qwen3-Next-80B -> Qwen-0.2B 蒸馏项目

这个目录包含一个可直接运行的蒸馏工程，目标是把 OpenRouter 提供的 `qwen/qwen3-next-80b-a3b-instruct:free` 的能力通过序列级蒸馏迁移到一个自定义的 `qwen-0.2B` 学生模型。

## 方案说明

大教师模型在普通 Windows 单机上通常无法本地加载，因此这里采用两阶段的可落地方案：

1. 用 OpenRouter 的 `qwen/qwen3-next-80b-a3b-instruct:free` 生成教师回复，产出伪标签数据集。
2. 用这些教师回复对 `qwen-0.2B` 学生模型做监督蒸馏训练。

这属于标准的 sequence-level distillation，优点是：

- Windows 环境可运行。
- 可以直接接 OpenAI 兼容接口的教师服务。
- 学生模型完全本地训练。

## 目录结构

- `configs/distill_qwen_0.2b.yaml`: 正式蒸馏配置
- `configs/smoke_test.yaml`: 本地烟雾测试配置
- `data/custom_prompts.example.jsonl`: 自定义提示词样例
- `data/smoke_distill.jsonl`: 本地训练连通性测试数据
- `scripts/setup_env.ps1`: 自动创建虚拟环境并安装依赖
- `scripts/train_distill.ps1`: 生成教师数据并训练学生模型
- `scripts/smoke_test.ps1`: 使用内置小样本验证训练入口
- `scripts/eval_quality.ps1`: 比较蒸馏前后回答质量

## 环境要求

- Windows PowerShell 5.1+
- Python 3.10 到 3.12
- 建议 CUDA 12.4 驱动环境；如果没有 GPU，也可以跑 `smoke test`

## 一键安装

在当前目录执行：

```powershell
Set-Location d:\Lenovo\github\ShowCase\distill_small_model
.\scripts\setup_env.ps1
```

脚本会自动完成：

- 创建 `.venv`
- 升级 `pip/setuptools/wheel`
- 检测 `nvidia-smi`
- 有 NVIDIA GPU 时安装 CUDA 版 `torch`
- 无 GPU 时安装默认 `torch`
- 安装本项目依赖

## 先做本地连通性验证

```powershell
Set-Location d:\Lenovo\github\ShowCase\distill_small_model
.\scripts\smoke_test.ps1
```

这个测试不会调用外部 API，也不会下载 Hugging Face tokenizer；它会基于 `data/smoke_distill.jsonl` 现场构建一个本地 tokenizer，然后训练几步，验证：

- 模型初始化
- tokenizer 加载
- dataset tokenization
- 手写训练循环
- checkpoint 写出

## 正式蒸馏前的配置

编辑 `configs/distill_qwen_0.2b.yaml`，重点关注：

- `teacher.api_base`
- `teacher.model_name`
- `teacher.api_key_env`
- `dataset.hf_dataset` 或 `dataset.local_prompts_path`
- `dataset.take_samples`
- `training.max_seq_length`
- `training.per_device_train_batch_size`
- `training.gradient_accumulation_steps`

默认配置使用 `yahma/alpaca-cleaned` 作为蒸馏提示集，也可以改成自己的 `jsonl` 文件。

如果你使用 OpenRouter 的免费教师模型，建议先把 `dataset.take_samples` 控制在 `50` 到 `200`，并把 `dataset.request_interval_seconds` 设到 `5` 秒以上。免费模型经常有上游限流，不适合一开始就跑大规模蒸馏。

## 配置教师 API Key

默认会读取环境变量 `OPENROUTER_API_KEY`：

```powershell
$env:OPENROUTER_API_KEY = "your_openrouter_api_key"
```

当前默认网关是 OpenRouter。如果你后续换到其他 OpenAI 兼容网关，只需要修改 `api_base`、`model_name` 和 `api_key_env`。

如果主教师模型经常返回 429，可以在 `teacher.fallback_model_names` 中配置备用模型。脚本会优先使用 `teacher.model_name`，失败后自动尝试备用模型。

## 开始正式蒸馏

```powershell
Set-Location d:\Lenovo\github\ShowCase\distill_small_model
.\scripts\train_distill.ps1
```

流程如下：

1. 从 Hugging Face 数据集或本地提示文件读取 prompts
2. 调用 `qwen/qwen3-next-80b-a3b-instruct:free` 生成教师回复
3. 保存到 `artifacts/datasets/openrouter_qwen3_next_80b_distill.jsonl`
4. 初始化 `qwen-0.2B` 学生模型
5. 在 `artifacts/checkpoints/qwen_0.2b_distill` 下开始训练并保存权重

## 比较蒸馏前后回答质量

先准备评估集。评估集是一个 `jsonl` 文件，每行至少包含 `prompt`，也可以额外提供 `reference`：

```json
{"prompt": "解释什么是 MoE 路由。", "reference": "MoE 路由会为每个 token 选择部分专家参与计算，从而在控制计算量的同时扩大参数规模。"}
```

然后执行：

```powershell
Set-Location d:\Lenovo\github\ShowCase\distill_small_model
.\scripts\eval_quality.ps1
```

默认会做两类比较：

- `baseline`: 蒸馏前学生模型
- `distilled`: 蒸馏后学生模型

如果评估集带有 `reference` 字段，脚本会计算离线指标：

- `rouge_l_f1`
- `token_f1`
- `exact_match`
- 输出长度统计

如果你在配置里开启 `evaluation.judge_with_teacher: true`，还会调用教师 API 做 pairwise 裁判，输出：

- `baseline_wins`
- `distilled_wins`
- `ties`

评估结果会写到：

- `artifacts/eval/generations.jsonl`
- `artifacts/eval/summary.json`

可以先运行离线 smoke 评估：

```powershell
Set-Location d:\Lenovo\github\ShowCase\distill_small_model
.\scripts\eval_quality.ps1 -Config configs/eval_smoke.yaml
```

## 使用自己的提示数据

如果你不想依赖 Hugging Face 数据集，可以准备一个 `jsonl` 文件，格式参考 `data/custom_prompts.example.jsonl`：

```json
{"prompt": "解释一下什么是 KV cache。"}
{"instruction": "写一个 Python 函数", "input": "要求计算斐波那契数列前 n 项"}
```

然后在配置里设置：

```yaml
dataset:
  local_prompts_path: ./data/your_prompts.jsonl
```

## 资源建议

- `0.2B` 学生模型可以在单卡上训练
- 如果只有 CPU，建议先跑 `smoke test` 验证流程，不建议直接正式训练
- 如果教师是远端 API，正式蒸馏的主要瓶颈通常是数据生成速度和费用

## 输出结果

- 学生初始化模型: `artifacts/student_init`
- 蒸馏数据集: `artifacts/datasets`
- 训练 checkpoint: `artifacts/checkpoints/qwen_0.2b_distill`
- 评估结果: `artifacts/eval`

## 已知边界

- 项目默认实现的是序列级蒸馏，不是在线 logits 蒸馏
- 如果你后续有多卡推理集群，可以再扩展成在线 KL 蒸馏
- `qwen-0.2B` 是本项目内定义的自定义 Qwen2 架构，不是官方发布型号
