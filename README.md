# Prover Finetune + Lean Eval Framework

一个面向数学形式化证明场景的 Python 项目，包含两条可配置工作流：

1. 使用 QLoRA 对小/中型 LLM 做微调。
2. 在 miniF2F 上做 Lean 证明实验，并用 Lean checker 自动验证结果。

---

## 功能概览

- **QLoRA 微调**：支持 Hugging Face 数据集和本地 JSONL，模板化构造训练文本。
- **miniF2F 实验评测**：支持从 `json` / `jsonl` / Lean 文件目录加载题目。
- **pass@k 评测**：单题可采样多个 proof 候选，任一通过 Lean 校验即记为通过。
- **模型模板自适应**：支持 `model_type` 自动识别，内置 DeepSeek-Prover-V2 专用提示模板。
- **LeanInteract 本地工程验证**：按本地 `mathlib4` 工程路径加载（`LocalProject`），不在代码中动态改写 Lean 工程。

---

## 项目结构

```text
configs/
  experiment.example.yaml      # 实验配置
  finetune.example.yaml        # 微调配置
  lean_project.example.yaml    # Lean 工程与 mathlib 配置

scripts/
  extract_minif2f_lean_to_json.py   # 解析 Valid.lean/Test.lean -> json
  analyze_numinamath_lean.py        # NuminaMath-LEAN 数据统计
  filter_numinamath_lean.py         # NuminaMath-LEAN 条件筛选导出

src/prover_finetune/
  experiments/
    run_experiment.py          # miniF2F 评测主入口
    lean_checker.py            # Lean 项目初始化与 proof 检查
    minif2f.py                 # miniF2F 多来源加载器
    prover.py                  # 模型推理封装
  finetune/
    train_qlora.py             # QLoRA 训练入口
    data.py                    # 数据加载与模板化处理
```

---

## 环境准备

### 1) Python 环境

```bash
conda create -n prover-finetune python=3.10 -y
conda activate prover-finetune
pip install -r requirements.txt
```

> 说明：`requirements.txt` 仅包含基础依赖，实际运行需有可用的 PyTorch/CUDA 环境（若进行 GPU 训练/推理）。

### 2) Lean / Lake 环境（手动）

请确保系统可执行 `lake` 命令（通常通过 `elan` 安装 Lean 工具链后可用）。  
本项目不再在代码中自动构建 Lean 工程，请先准备本地 `mathlib4` 目录（推荐子模块）并手动执行 `lake`。

---

## 工作流 A：miniF2F 实验评测

### 0) 准备 mathlib4（支持多版本并存）

```bash
python scripts/setup_mathlib4.py \
  --config configs/lean_project.example.yaml \
  --sync-mathlib-path
```

目录名默认包含版本号（来自 `mathlib_setup.dir_template`，默认 `mathlib4-{ref}`），例如 `external/mathlib4-v4.27.0`。

如果你要并存多个版本，可以复制一份 Lean 配置并改 `mathlib_setup.ref`：

```bash
cp configs/lean_project.example.yaml configs/lean_project.v4.28.0.yaml
# 编辑 configs/lean_project.v4.28.0.yaml 中的 mathlib_setup.ref = v4.28.0
python scripts/setup_mathlib4.py --config configs/lean_project.v4.28.0.yaml --sync-mathlib-path
```

### 1) 下载 miniF2F Lean 数据

```bash
mkdir -p data/minif2f
curl -L "https://codeload.github.com/google-deepmind/miniF2F/zip/refs/heads/main" -o "data/minif2f/google-deepmind-miniF2F.zip"
unzip -o "data/minif2f/google-deepmind-miniF2F.zip" -d data/minif2f
```

解压后常见路径：

- `data/minif2f/miniF2F-main/MiniF2F/Valid.lean`
- `data/minif2f/miniF2F-main/MiniF2F/Test.lean`

### 2) 解析为 JSON

```bash
python scripts/extract_minif2f_lean_to_json.py
```

输出文件：

- `data/processed/valid.json`
- `data/processed/test.json`

### 3) 配置实验

编辑 `configs/experiment.example.yaml`：

- `experiment`
  - `split`: `valid` / `test`
  - `max_samples`: 调试时可先设小值
  - `pass_k`: 每题最多尝试的候选 proof 数（用于 pass@k）
  - `gpu_ids`: 并行评测使用的 GPU 列表（必填，例如 `[0]` 或 `[0,1,2]`）
  - `verbose_logging`: 是否输出详细中间日志（完整生成、Lean 检查输出、重试进度）
  - `output_dir`: 实验输出目录
- `model`
  - `name_or_path`: 基座模型
  - `model_type`: `auto` / `generic` / `deepseek_prover_v2`
  - `adapter_path`: LoRA 适配器目录（可为空）
  - 设备参数：`gpu_device`（如 `0` 或 `cuda:1`）/ `device_map`（更底层，优先级更高）
  - 生成参数：`inference_timeout_sec` / `max_new_tokens` / `temperature` / `top_p` / `do_sample`
- `minif2f`
  - `source_type`: `local_json` / `local_jsonl` / `local_lean_dir`
  - 对应设置 `json_path`、`jsonl_path` 或 `root_dir + split_file`
- `lean`
  - `project_config_path`: 指向 `configs/lean_project.example.yaml`

再编辑 `configs/lean_project.example.yaml`（参考 `test/lean_verifier.py` 用法）：

- 本地工程路径：`mathlib_path`（例如 `external/mathlib4-v4.27.0`）
- Lean 版本：`lean_version`（默认 `v4.27.0`；若 `mathlib_path/lean-toolchain` 存在会自动读取该版本）
- 运行时：`timeout_sec`
- lean-interact 模式：`use_lean_interact`、`use_auto_server`、`memory_limit_mb`
- 校验头部：`header_imports`、`header_set_options`、`header_open_scopes`

### 4) 运行实验

```bash
python -m src.prover_finetune.experiments.run_experiment --config configs/experiment.example.yaml
```

### 5) 查看结果

输出目录（默认示例：`outputs/experiments/minif2f-baseline`）中包含：

- `summary.json`：整体统计（`total`、`pass`、`pass@k`）
- `results.json`：逐题结果（首个预测、候选列表、Lean 输出日志等）
- `experiment.log`：整体运行日志
- `problem_logs/<name>.log`：每道题独立日志（优先用题目 `name` 命名，缺失时回退到 `id`）

并行执行说明：

- 会按 `gpu_ids` 长度启动同等数量线程（每线程绑定一个 GPU）。
- 每线程独立初始化 prover 与 Lean checker，避免 `Main.lean` 互相覆盖。
- 若 `gpu_ids` 包含不可见设备编号，程序会在启动时直接报错。

`results.json` 中每条样本会记录：

- `prediction`：第一个候选 proof（便于快速查看）
- `candidates`：按采样顺序保存所有候选的 `ok`、`prediction`、`lean_output`
- 一旦某候选校验通过，会提前停止该题后续候选验证（提高评测速度）

### 6) DeepSeek-Prover-V2 pass@k 示例

项目已提供配置：

- `configs/experiment.deepseek_prover_v2_7b.pass16.yaml`

运行方式：

```bash
python -m src.prover_finetune.experiments.run_experiment \
  --config configs/experiment.deepseek_prover_v2_7b.pass16.yaml
```

---

## 工作流 B：QLoRA 微调

### 1) 配置微调参数

编辑 `configs/finetune.example.yaml`：

- `model`
  - `name_or_path`、`trust_remote_code`
  - `use_4bit` / `use_8bit`（二选一）
  - `lora`（`r`、`alpha`、`dropout`、`target_modules`）
- `data`
  - `source_type`: `huggingface` 或 `jsonl`
  - `formatter_type`: `auto` / `generic` / `deepseek_prover_v2`
  - Hugging Face: `dataset_name`、`dataset_config`、`train_split`（`eval_split` 可选）
  - JSONL: `train_path`、`eval_path`
  - 通用模板：`template`（可直接使用原始字段，如 `{formal_ground_truth}`）
  - DeepSeek 模板字段：`formal_statement_field`、`reasoning_steps_field`、`proof_field`
  - `max_seq_length`
- `training`
  - `output_dir`、batch size、学习率、warmup、scheduler、save/eval steps 等

### 2) 启动训练

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.prover_finetune.finetune.train_qlora --config configs/finetune.example.yaml
```

训练完成后会将 adapter 与 tokenizer 保存到：

- `<output_dir>/adapter`

当 `formatter_type=deepseek_prover_v2` 时，训练样本会被格式化为：

- 用户部分：给定 theorem 的 DeepSeek-Prover-V2 风格 prompt
- 助手部分：`Proof plan` + `Lean 4 code`（代码块）

---

## NuminaMath-LEAN 数据分析与筛选

项目提供 `scripts/analyze_numinamath_lean.py` 用于统计 `AI-MO/NuminaMath-LEAN` 数据集的 token 分布：

```bash
python scripts/analyze_numinamath_lean.py \
  --dataset-name AI-MO/NuminaMath-LEAN \
  --tokenizer-name gpt2
```

该脚本会输出：

- 数据过滤后样本数（默认过滤 `author=human` 且 `ground_truth_type=complete`）
- `formal_ground_truth` token 长度分桶分布（`0-256` 到 `>8192`）
- 最长样本 token 数与索引
- 最短样本 token 数、索引，以及对应 `problem` 与 `formal_ground_truth`

还提供筛选导出脚本 `scripts/filter_numinamath_lean.py`，用于生成训练数据：

```bash
python scripts/filter_numinamath_lean.py \
  --dataset-name AI-MO/NuminaMath-LEAN \
  --tokenizer-name gpt2 \
  --max-formal-tokens 4096
```

默认筛选条件：

- `author == human`
- `ground_truth_type == complete`
- `formal_ground_truth` token 数 `<= 4096`

默认输出文件：

- `data/processed/numinamath_lean_filtered_train.jsonl`

输出 JSONL 会额外包含：

- `formal_ground_truth_token_count`（便于后续按长度做分桶训练）

---

## 测试（数据格式）

可运行数据格式测试，验证 DeepSeek 数据模板拼接逻辑：

```bash
python test/test_data_formatting.py
```

该测试主要检查：

- `deepseek_prover_v2` 格式是否包含 `Proof plan` 与 `Lean 4 code`
- `load_and_process_dataset` 是否正确产出 `text` 字段

---

## 常见问题

- **`lake: command not found`**
  - 说明 Lean 工具链未安装或未加入 PATH，请先安装 `elan/lean`。
- **验证器提示找不到 `mathlib_path`**
  - 请先运行 `scripts/setup_mathlib4.py`，并确认配置里的 `mathlib_path` 指向已构建目录。
- **模型显存不足**
  - 可降低 `max_new_tokens`、换更小模型、启用 4bit 量化，或减小 batch。
- **生成 proof 通过率低**
  - 先提高 prompt 约束、增大训练数据质量，再针对特定题型做微调。

---

## 备注

- `miniF2F` 加载器默认会按 `split` 过滤；若数据行不含 `split` 字段，会直接纳入当前评测集合。
- Lean 校验时会将 theorem 与模型生成 proof 写入 `Main.lean` 并执行 `lake env lean Main.lean`。
- `.gitignore` 默认忽略 `data/**`，但保留 `data/processed/**` 便于版本化关键处理结果。
- 详细 Lean/Mathlib 版本管理可参考 `docs/lean-mathlib-config.md`（若你本地已维护该文档）。
