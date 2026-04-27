# Prover Finetune + Lean Eval Framework

一个面向数学形式化证明场景的 Python 项目，包含两条可配置工作流：

1. 使用 QLoRA 对小/中型 LLM 做微调。
2. 在 miniF2F 上做 Lean 证明实验，并用 Lean checker 自动验证结果。

---

## 功能概览

- **QLoRA 微调**：支持 Hugging Face 数据集和本地 JSONL，模板化构造训练文本。
- **miniF2F 实验评测**：支持从 `json` / `jsonl` / Lean 文件目录加载题目。
- **Lean 环境自动初始化**：首次运行实验时自动生成 `lean-toolchain`、`lakefile.lean`，并拉取 mathlib 缓存。
- **可复用的版本戳机制**：当 Lean/Mathlib 配置不变时跳过重复构建。

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

### 2) Lean / Lake 环境

请确保系统可执行 `lake` 命令（通常通过 `elan` 安装 Lean 工具链后可用）。  
实验首次运行会在配置的 `project_dir` 下自动生成和更新 Lean 工程。

---

## 工作流 A：miniF2F 实验评测

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
  - `output_dir`: 实验输出目录
- `model`
  - `name_or_path`: 基座模型
  - `adapter_path`: LoRA 适配器目录（可为空）
  - 生成参数：`max_new_tokens` / `temperature` / `top_p` / `do_sample`
- `minif2f`
  - `source_type`: `local_json` / `local_jsonl` / `local_lean_dir`
  - 对应设置 `json_path`、`jsonl_path` 或 `root_dir + split_file`
- `lean`
  - `project_config_path`: 指向 `configs/lean_project.example.yaml`

再编辑 `configs/lean_project.example.yaml`：

- 版本与依赖：`lean_version`、`mathlib_ref`、`extra_dependencies`
- 运行时：`timeout_sec`、`cache_get_timeout_sec`
- 校验头部：`header_imports`、`header_set_options`、`header_open_scopes`

### 4) 运行实验

```bash
python -m src.prover_finetune.experiments.run_experiment --config configs/experiment.example.yaml
```

### 5) 查看结果

输出目录（默认示例：`outputs/experiments/minif2f-baseline`）中包含：

- `summary.json`：整体统计（`total`、`pass`、`pass@1`）
- `results.json`：逐题结果（预测 proof、Lean 输出日志等）

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
  - Hugging Face: `dataset_name`、`dataset_config`、`train_split`、`eval_split`
  - JSONL: `train_path`、`eval_path`
  - 文本模板：`template`（支持 `{prompt}`、`{completion}` 和原始字段）
  - `max_seq_length`
- `training`
  - `output_dir`、batch size、学习率、warmup、scheduler、save/eval steps 等

### 2) 启动训练

```bash
python -m src.prover_finetune.finetune.train_qlora --config configs/finetune.example.yaml
```

训练完成后会将 adapter 与 tokenizer 保存到：

- `<output_dir>/adapter`

---

## 数据分析脚本（可选）

项目提供 `scripts/analyze_numinamath_lean.py` 用于统计 `AI-MO/NuminaMath-LEAN` 数据集的 token 分布：

```bash
python scripts/analyze_numinamath_lean.py \
  --dataset-name AI-MO/NuminaMath-LEAN \
  --split train \
  --tokenizer-name gpt2
```

该脚本会输出：

- 数据过滤后样本数（默认过滤 `author=human` 且 `ground_truth_type=compete`）
- `formal_ground_truth` token 长度分桶分布
- 最长样本 token 数与索引

---

## 常见问题

- **`lake: command not found`**
  - 说明 Lean 工具链未安装或未加入 PATH，请先安装 `elan/lean`。
- **首次实验启动较慢**
  - 正常现象：会执行 `lake update` 与 `lake exe cache get` 拉取依赖与缓存。
- **模型显存不足**
  - 可降低 `max_new_tokens`、换更小模型、启用 4bit 量化，或减小 batch。
- **生成 proof 通过率低**
  - 先提高 prompt 约束、增大训练数据质量，再针对特定题型做微调。

---

## 备注

- `miniF2F` 加载器默认会按 `split` 过滤；若数据行不含 `split` 字段，会直接纳入当前评测集合。
- Lean 校验时会将 theorem 与模型生成 proof 写入 `Main.lean` 并执行 `lake env lean Main.lean`。
- 详细 Lean/Mathlib 版本管理可参考 `docs/lean-mathlib-config.md`（若你本地已维护该文档）。
