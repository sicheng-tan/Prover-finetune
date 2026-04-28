# AutoFormal

自动化形式化和定理证明流水线：自然语言 → Lean 4 形式化 → 自动证明

## 核心功能

- 🤖 自动形式化 (NL → Lean 4)
  - **错误信息利用机制**：自动利用 Lean 验证错误和一致性检查错误改进生成
  - **自我修复机制**：类似 prover，支持多次修复尝试（可配置修复轮次）
  - **增强日志系统**：详细记录每次生成、验证和修复过程
- 🔬 自动证明生成 + 自我纠错
  - **Cursor CLI 额外修复机制**：在 prover 的 self-correction 失败后，可选的 Cursor CLI 修复机制（使用 Cursor headless CLI）
- ✅ Lean 4 编译器验证
- 🔀 可选问题转换 (非证明问题 → 证明问题)
- 🔄 可选形式化验证 (Lean → NL 反向验证)
- ☁️ 支持云端 API (GPT-4, DeepSeek-v3)
- 🚀 支持 Mock 模式测试
- 📦 批量处理 + 断点续传
- 🔄 **多 Prover 支持**：配置多个证明模型，按顺序尝试直到成功
- 🎯 **直接证明已有Statement**：使用已有的形式化statement直接调用prover，跳过formalization步骤

## 快速开始

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 下载 mathlib4 (使用 git submodules)
git submodule update --init --recursive

# 3. 配置 mathlib4 (使用 lake)
cd mathlib4
lake exe cache get  # 获取预编译文件（可选但强烈推荐，可大幅加快构建速度）
lake build
cd ..

# 4. 测试 (无需模型)
python test/test_mock_mode.py

# 5. 运行 - 三种模式
python run_pipeline.py --mock          # Mock: 测试用
python run_pipeline.py                 # 根据配置文件选择模式（本地模型或 OpenAI API）
# 注意：--openai 参数已弃用，推荐在配置文件中直接配置 OpenAI 模型
```

## 三种运行模式

| 模式 | 启动 | 速度 | GPU | 磁盘 | 成本 | 真实证明 |
|------|------|------|-----|------|------|---------|
| Mock | < 5s | 0.01s | ❌ | < 1GB | 免费 | ❌ |
| OpenAI | < 5s | 5-30s | ❌ | < 1GB | API费用 | ✅ |
| Full | 2-5min | 60-600s | 16GB+ | 40GB | 硬件 | ✅ |

## 使用方法

### 1. OpenAI API 模式 (推荐 - 无需 GPU)

**新方式（推荐）**：直接在配置文件中使用 `model` 字段

```bash
# 设置 API key
export OPENAI_API_KEY="your-key"

# 运行（无需 --openai 参数）
python run_pipeline.py --start 0 --end 10
```

**配置** (`config/config.yaml`):
```yaml
# 不需要设置 openai_mode: true
formalizer:
  model: "deepseek-chat"  # 使用 'model' 而不是 'model_id' 表示使用 OpenAI API
  base_url: "https://api.deepseek.com/v1"  # 可选：自定义 API 端点
  max_tokens: 16384
  temperature: 0.9
  top_p: 0.95

provers:
  - model: "deepseek-chat"  # 使用 'model' 而不是 'model_id'
    base_url: "https://api.deepseek.com/v1"
    max_tokens: 32768
    temperature: 0.7
    top_p: 0.95

openai:
  api_key: ""  # 或使用环境变量 OPENAI_API_KEY
```

**旧方式（已弃用，但仍支持）**：使用 `--openai` 参数或 `openai_mode: true`
```bash
python run_pipeline.py --openai --start 0 --end 10
```
```yaml
openai_mode: true  # 已弃用，仅用于向后兼容
openai:
  api_key: ""
  formalizer:
    model: "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"
  provers:
    - model: "deepseek-chat"
      base_url: "https://api.deepseek.com/v1"
```

### 2. 本地模型模式

```bash
# 首次运行会自动下载模型 (~32GB)
python run_pipeline.py --start 0 --end 10
```

### 3. Python API

```python
from src.pipeline import AutoFormalPipeline

# 选择模式（根据配置文件自动判断）
pipeline = AutoFormalPipeline(mock_mode=True)    # Mock 模式
pipeline = AutoFormalPipeline()                  # 根据配置文件选择（本地模型或 OpenAI API）

# 注意：openai_mode=True 已弃用，推荐在配置文件中直接配置 OpenAI 模型
# 如果配置文件中 formalizer 和 provers 使用 'model' 字段，会自动使用 OpenAI API

# 处理问题
result = pipeline.process_single_problem(
    problem_id="test",
    nl_problem="Prove that 2 + 2 = 4"
)

# 批量处理
results = pipeline.run(
    input_file="data/practice_data.jsonl",
    output_file="results.jsonl",
    start_idx=0,
    end_idx=10
)
```

### 4. 直接证明已有形式化Statement

如果您已经有了形式化的Lean 4 statement，可以直接调用prover，跳过formalization步骤：

**方法1: 使用 `prove_existing_statement` 方法**

```python
from src.pipeline import AutoFormalPipeline

pipeline = AutoFormalPipeline(config_path="config/config.yaml")

result = pipeline.prove_existing_statement(
    problem_id="example_001",
    formal_statement="theorem problem_example_001 : 2 + 2 = 4 := by sorry",
    header="import Mathlib.Data.Real.Basic\nimport Mathlib.Tactic",
    nl_problem="Prove that 2 + 2 = 4",  # 可选
    output_file="data/result.jsonl"  # 可选
)
```

**方法2: 在problem字典中提供已有statement**

```python
# 方式A: 提供formal_statement和header
problem = {
    "id": "example_001",
    "nl_problem": "Prove that 2 + 2 = 4",
    "formal_statement": "theorem problem_example_001 : 2 + 2 = 4 := by sorry",
    "header": "import Mathlib.Data.Real.Basic\nimport Mathlib.Tactic"
}

# 方式B: 提供完整的formal_code
problem = {
    "id": "example_001",
    "nl_problem": "Prove that 2 + 2 = 4",
    "formal_code": """import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem problem_example_001 : 2 + 2 = 4 := by sorry"""
}

result = pipeline.process_problem(problem)
```

**支持的字段**：
- `formal_statement`: 形式化statement（theorem/lemma/def部分）
- `header`: 可选的header（imports, namespace等）
- `formal_code` 或 `full_code`: 完整的Lean代码（包含header和statement）

如果提供了这些字段，pipeline会：
1. 跳过formalization步骤
2. 直接使用提供的statement调用prover
3. 在结果中标记 `formalization_skipped: true`

**使用场景**：
- 已有形式化结果，只需要进行证明
- 从JSONL文件读取已有形式化statement，批量进行证明
- 调试和测试prover功能
- 增量处理：formalization成功但proof失败后，重新尝试证明

**方法3: 使用命令行接口**

**仅证明模式**：从JSONL文件证明已有statement（支持单个或批量）：
```bash
# 证明单个问题（第一个，或使用--start指定索引）
python run_pipeline.py --prove-only \
    --input data/formalized_statements.jsonl \
    --output data/proof_results.jsonl \
    --openai

# 批量证明（处理所有问题）
# 推荐：在配置文件中配置 OpenAI 模型，无需 --openai 参数
python run_pipeline.py --prove-only \
    --input data/formalized_statements.jsonl \
    --output data/proof_results.jsonl

# 指定范围
python run_pipeline.py --prove-only \
    --input data/formalized_statements.jsonl \
    --start 0 --end 10 \
    --output data/proof_results.jsonl
```

JSONL文件应包含 `id`、`formal_statement`（或`formal_code`），以及可选的`header`和`nl_problem`字段。

**仅形式化模式**：从JSONL文件形式化自然语言问题（支持单个或批量）：
```bash
# 形式化所有问题
# 推荐：在配置文件中配置 OpenAI 模型，无需 --openai 参数
python run_pipeline.py --formalizer-only \
    --input data/practice_data.jsonl \
    --output data/formalized_only.jsonl

# 形式化指定范围
python run_pipeline.py --formalizer-only \
    --input data/practice_data.jsonl \
    --start 0 --end 10 \
    --output data/formalized_only.jsonl
```

JSONL文件应包含 `id` 和 `nl_problem` 字段。

**示例脚本**：
```bash
python test/example_prove_existing_statement.py
```

## 配置文件

`config/config.yaml`:

**新方式（推荐）**：直接在 `formalizer` 和 `provers` 中使用 `model` 字段配置 OpenAI API

```yaml
# 模式选择
mock_mode: false      # 测试模式
# openai_mode: false  # 已弃用，不再需要设置

# Formalizer 配置：使用 'model' 表示 OpenAI API，使用 'model_id' 表示本地模型
formalizer:
  # OpenAI API 示例（推荐）
  model: "deepseek-chat"  # 或 "gpt-4", "gpt-4-turbo" 等
  base_url: "https://api.deepseek.com/v1"  # 可选：自定义 API 端点
  max_tokens: 16384
  temperature: 0.9
  top_p: 0.95
  max_retries: 5
  correction_rounds: 1
  
  # 本地模型示例（注释掉上面的 OpenAI 配置，取消注释下面的配置）
  # model_id: "Goedel-LM/Goedel-Formalizer-V2-8B"
  # max_new_tokens: 16384
  # device: "cuda:0"  # GPU 设备：cuda:0, cuda:1, cuda:2, cuda:3 或 "auto"
  
  # 问题转换（可选）
  # 将非证明问题（如"What is...?"）转换为证明问题
  enable_problem_conversion: false  # 启用问题转换
  
  # 形式化验证（可选）
  # 注意：仅在 Lean 验证通过后才执行
  enable_informalization_check: false  # 启用 Lean → NL 反向验证

# Prover 设置（支持多个 prover 按顺序尝试）
# 可以混合使用 OpenAI API 模型和本地模型
provers:
  # OpenAI API 示例（推荐）
  - model: "deepseek-chat"  # 使用 'model' 表示 OpenAI API
    base_url: "https://api.deepseek.com/v1"
    max_tokens: 32768
    temperature: 0.7
    top_p: 0.95
    max_retries: 3
    self_correction_rounds: 2
  
  # 本地模型示例
  # - model_id: "Goedel-LM/Goedel-Prover-V2-8B"  # 使用 'model_id' 表示本地模型
  #   max_new_tokens: 32768
  #   device: "cuda:1"  # GPU 设备：cuda:0, cuda:1, cuda:2, cuda:3 或 "auto"
  #   timeout: null  # 超时时间（秒），null 表示不限制
  
  # 可选：添加更多 prover，当前面的失败时会依次尝试
  # - model_id: "deepseek-ai/DeepSeek-Prover-V2-7B"
  #   max_retries: 3
  #   self_correction_rounds: 2
  #   device: "cuda:2"  # 可以使用不同的 GPU
  #   timeout: null

# Cursor CLI Corrector（可选，额外的代码修复机制）
# 当启用时，在 prover 的 self-correction 失败后，会调用 Cursor CLI 来尝试修复 Lean 代码
# 这使用 Cursor 的 headless CLI 模式（cursor-agent）
# 参考文档：https://cursor.com/docs/cli/overview
prover_cursor_corrector:
  enable: false  # 设置为 true 启用 Cursor CLI corrector
  api_key: ""  # Cursor API key（也可以使用 CURSOR_API_KEY 环境变量）
  cursor_cli_path: null  # cursor-agent 可执行文件路径（默认：PATH 中的 "cursor-agent"）
  model: null  # 使用的模型（可选，不指定则使用 Cursor 默认）
  max_correction_attempts: 2  # 每次调用的最大修复尝试次数
  temp_dir: null  # 代码文件的临时目录（默认：系统临时目录）
  mathlib_path: null  # mathlib4 目录路径（默认：使用 lean.mathlib_path）
  timeout_per_attempt: 300  # 每次 Cursor CLI 调用的超时时间（秒，默认：300秒/5分钟）
  total_timeout: null  # 所有修复尝试的总超时时间（秒，默认：null，无限制）

# OpenAI API 设置（仅用于提供 API key）
# 注意：推荐在 formalizer 和 provers 中直接使用 'model' 字段配置 OpenAI 模型
# 下面的 openai.formalizer 和 openai.provers 仅用于向后兼容（当 openai_mode: true 时）
openai:
  api_key: ""  # 或使用环境变量 OPENAI_API_KEY
  # 以下配置已弃用，推荐在 formalizer 和 provers 中直接配置
  # formalizer:
  #   model: "deepseek-chat"
  #   base_url: "https://api.deepseek.com/v1"
  # provers:
  #   - model: "deepseek-chat"
  #     base_url: "https://api.deepseek.com/v1"

# Lean 设置
lean:
  timeout: 300
  mathlib_path: "mathlib4"
  lean_version: "v4.25.0-rc2"  # Lean 版本（可选，默认从 mathlib4/lean-toolchain 读取）
  use_auto_server: true  # 使用 AutoLeanServer（更好的错误恢复）
  verbose: false  # 详细日志
```

**Lean 版本说明**：
- 如果 `mathlib4/lean-toolchain` 存在，会自动使用其中的版本（推荐）
- 否则使用配置文件中的 `lean_version`
- 默认值：`v4.25.0-rc2`

### 多 Prover 支持

Pipeline 支持配置多个证明模型，按顺序尝试直到某个成功。这在以下场景很有用：
- 不同模型擅长不同类型的问题
- 提供备用模型以提高成功率
- 在不同 GPU 上分配模型以优化资源使用

**配置示例**：

```yaml
provers:
  # 第一个 prover：Goedel-Prover
  - model_id: "Goedel-LM/Goedel-Prover-V2-8B"
    max_retries: 3
    self_correction_rounds: 2
    device: "cuda:1"
    timeout: 600  # 10 分钟超时
  
  # 第二个 prover：DeepSeek-Prover（如果第一个失败）
  - model_id: "deepseek-ai/DeepSeek-Prover-V2-7B"
    max_retries: 3
    self_correction_rounds: 2
    device: "cuda:2"  # 使用不同的 GPU
    timeout: 600
```

**工作流程**：
1. 使用第一个 prover 尝试证明
2. 如果失败，自动切换到第二个 prover
3. 继续尝试直到某个 prover 成功或全部失败
4. 结果中会记录所有尝试和成功的 prover

**输出示例**：

```json
{
  "proof_success": true,
  "successful_prover": "deepseek-ai/DeepSeek-Prover-V2-7B",
  "prover_attempts": [
    {
      "prover_index": 0,
      "prover_name": "Goedel-LM/Goedel-Prover-V2-8B",
      "success": false,
      "attempts": 3,
      "error": "Max retries exceeded"
    },
    {
      "prover_index": 1,
      "prover_name": "deepseek-ai/DeepSeek-Prover-V2-7B",
      "success": true,
      "attempts": 2,
      "corrections": 1
    }
  ]
}
```

**OpenAI API 模式也支持多 Prover**（推荐方式）：

```yaml
# 不需要设置 openai_mode: true
provers:
  - model: "gpt-4"  # 使用 'model' 表示 OpenAI API
    max_tokens: 32768
    temperature: 0.7
    top_p: 0.95
    max_retries: 3
    self_correction_rounds: 2
  - model: "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"
    max_tokens: 32768
    temperature: 0.7
    top_p: 0.95
    max_retries: 3
    self_correction_rounds: 2

openai:
  api_key: ""  # 或使用环境变量 OPENAI_API_KEY
```

**旧方式（已弃用，但仍支持）**：
```yaml
openai_mode: true  # 已弃用
openai:
  api_key: ""
  provers:
    - model: "gpt-4"
      max_retries: 3
      self_correction_rounds: 2
    - model: "deepseek-chat"
      base_url: "https://api.deepseek.com/v1"
      max_retries: 3
      self_correction_rounds: 2
```

**向后兼容**：旧的单 prover 配置仍然支持：

```yaml
# 旧格式（仍然有效）
prover:
  model_id: "Goedel-LM/Goedel-Prover-V2-8B"
  max_retries: 3
  device: "cuda:1"
```

### Cursor CLI Corrector（额外修复机制）

当 prover 的 self-correction 机制失败后，可以启用 Cursor CLI corrector 作为额外的修复机制。这个功能使用 Cursor 的 headless CLI 模式（`cursor-agent`）来自动修复失败的 Lean 代码。

**前置要求**：
1. 安装 Cursor CLI：确保 `cursor-agent` 在 PATH 中或指定路径
2. 设置 API key：通过环境变量 `CURSOR_API_KEY` 或配置文件设置

**工作流程**：
1. Prover 生成证明代码
2. 如果验证失败，进行 self-correction 轮次
3. 如果所有 self-correction 都失败，**且启用了 Cursor corrector**，则调用 Cursor CLI 进行修复
4. Cursor CLI 分析错误信息并修改临时文件中的代码
5. 读取修复后的代码并验证

**配置示例**：

```yaml
# 启用 Cursor CLI corrector
prover_cursor_corrector:
  enable: true
  api_key: ""  # 或使用 CURSOR_API_KEY 环境变量
  cursor_cli_path: null  # 可选：指定 cursor-agent 路径
  model: null  # 可选：指定模型
  max_correction_attempts: 2
  temp_dir: null  # 可选：指定临时目录
  mathlib_path: null  # 可选：指定 mathlib4 路径（默认使用 lean.mathlib_path）
  timeout_per_attempt: 300  # 每次调用的超时时间（秒，默认：300秒）
  total_timeout: 600  # 所有尝试的总超时时间（秒，默认：null，无限制）
  log_detailed_output: false  # 如果为 true，记录 Cursor CLI 的关键信息（返回码、错误信息等）（默认：false）
```

**详细输出选项**：
- `log_detailed_output`: 设置为 `true` 时，会在日志中记录关键信息，包括：
  - Cursor CLI 的返回码（成功/失败）
  - 错误信息（如果有）
  - 每次尝试的基本信息（输入代码长度、错误信息长度等）
  - 验证结果（通过/失败）

**超时控制**：
- `timeout_per_attempt`: 每次调用 Cursor CLI 的最大等待时间（默认：300秒/5分钟）
- `total_timeout`: 所有修复尝试的总时间限制（默认：无限制）
  - 如果设置了总超时，会在每次尝试前检查剩余时间
  - 如果总时间用尽，会立即停止并返回失败
  - 这可以防止修复过程无限运行

**Mathlib 路径说明**：
- Cursor corrector 会在 prompt 中提供 mathlib4 的路径信息
- 这样 Cursor 可以查找 mathlib 中的定理、定义和策略
- 默认使用与 `lean.mathlib_path` 相同的路径（通常是 "mathlib4"）
- 如果指定了 `mathlib_path`，则使用指定的路径

**详细输出选项**：
- `log_detailed_output`: 设置为 `true` 时，会在日志中记录关键信息，包括：
  - Cursor CLI 的返回码（成功/失败）
  - 错误信息（如果有）
  - 每次尝试的基本信息（输入代码长度、错误信息长度等）
  - 验证结果（通过/失败）

**使用场景**：
- Prover 的 self-correction 无法修复某些类型的错误
- 需要利用 Cursor 的强大代码修复能力
- 作为最后的修复尝试，提高整体成功率
- 需要调试修复过程时，启用详细输出选项

**输出示例**：

```json
{
  "proof_success": true,
  "prover_attempts": [
    {
      "prover_index": 0,
      "prover_name": "Goedel-LM/Goedel-Prover-V2-8B",
      "success": true,
      "attempts": 3,
      "corrections": 2,
      "cursor_corrector_used": true,
      "cursor_corrector_attempts": 1
    }
  ]
}
```

**注意**：
- 启用此功能需要 Cursor API key（通过 `CURSOR_API_KEY` 环境变量或配置文件）
- Cursor corrector 会在所有 self-correction 轮次失败后才被调用
- 默认进行 2 次 Cursor CLI 修复尝试
- 在 Mock 模式下，Cursor corrector 会被自动禁用
- 需要确保 `cursor-agent` 命令可用（在 PATH 中或通过 `cursor_cli_path` 指定）

**安装 Cursor CLI**：
参考 [Cursor CLI 文档](https://cursor.com/docs/cli/overview) 安装 `cursor-agent`。

### 问题转换 (Problem Conversion)

这是一个可选的预处理步骤，将非证明问题（如计算问题）自动转换为证明问题。

**示例**:
- 输入: "What is the sum of all multiples of 3 between 100 and 200?"
- 转换后: "Prove that the sum of all multiples of 3 between 100 and 200 equals 4950."

```yaml
formalizer:
  enable_problem_conversion: true  # 启用问题转换（自动启用代码生成和执行）

openai:
  problem_converter:
    model: "gpt-4"  # 或 "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"  # 可选
    max_tokens: 2048
    temperature: 0.3
    code_execution_timeout: 30.0  # 代码执行超时时间（秒）
    max_code_gen_retries: 3  # 代码生成最大重试次数
    max_code_exec_retries: 3  # 代码执行最大重试次数（带错误反馈）
```

**注意**: 
- 启用此功能需要 OpenAI API key（即使在本地模型模式下）
- 当 `enable_problem_conversion: true` 时，代码生成和执行会自动启用
- 代码执行结果会直接作为补充信息传递给 formalizer，不再进行问题转换
- 模型会自动判断是否需要代码执行

### 形式化器增强功能

#### 错误信息利用机制

Formalizer 现在能够自动利用验证过程中的错误信息来改进生成：

- **Lean 验证错误**：当 Lean 编译器报告错误时，错误信息会被传递给下一次生成尝试
- **一致性检查错误**：当一致性检查失败时，相似度分数、解释等信息会被用于改进
- **代码提取失败**：当无法从模型输出中提取代码时，会提供明确的错误反馈

**工作流程**：
1. 生成形式化代码
2. 如果验证失败，收集错误信息
3. 在下次尝试时，将错误信息和失败代码作为上下文传递给模型
4. 模型根据错误反馈生成修正版本

#### 自我修复机制

Formalizer 现在支持类似 Prover 的自我修复机制：

- **双层循环结构**：
  - 外层循环：`max_retries` 次初始尝试
  - 内层循环：每次失败后进行 `correction_rounds` 次修复尝试
- **可配置修复轮次**：通过 `correction_rounds` 参数控制（默认: 1）
- **错误反馈传递**：每次修复都使用前一次的错误信息

**配置示例**：
```yaml
formalizer:
  max_retries: 5  # 最多 5 次初始尝试
  correction_rounds: 2  # 每次失败后尝试 2 次修复
```

**工作流程**：
```
尝试 1:
  - 生成代码 → 验证失败
  - 修复轮次 1: 使用错误信息生成新代码 → 验证失败
  - 修复轮次 2: 使用错误信息生成新代码 → 验证成功 ✓
  - 返回成功（attempts: 1, corrections: 2）
```

#### 增强日志系统

Formalizer 现在提供详细的日志输出，包括：

- **每次尝试的完整流程**：生成、验证、修复
- **原始模型输出**：记录模型生成的原始文本
- **提取的代码**：记录从输出中提取的 Lean 代码
- **验证结果**：详细的验证信息和错误消息
- **修复过程**：记录每次修复尝试和结果

日志使用清晰的分隔线和符号标记，便于调试和追踪问题。

### 形式化验证 (Informalization Check)

这是一个可选的质量检查步骤，通过将 Lean 语句翻译回自然语言并与原问题比较，验证形式化的语义正确性。

**重要**: 此验证仅在 Lean 编译器验证通过后才执行，用于检查语义正确性。

```yaml
formalizer:
  enable_informalization_check: true  # 启用验证

openai:
  informalizer:
    model: "gpt-4"  # 或 "deepseek-chat"
    base_url: "https://api.deepseek.com/v1"  # 可选
    similarity_threshold: 0.7  # 保留用于向后兼容（不再使用）
```

**注意**: 
- 启用此功能需要 OpenAI API key（即使在本地模型模式下）
- 验证基于模型的 YES/NO 判断，不再使用分数阈值

## 输入输出格式

**输入** (`data/practice_data.jsonl`):
```json
{"id": "001", "nl_problem": "Prove that 2 + 2 = 4"}
```

**输入（使用已有形式化statement）**:
```json
{
  "id": "001",
  "nl_problem": "Prove that 2 + 2 = 4",
  "formal_statement": "theorem problem_001 : 2 + 2 = 4 := by sorry",
  "header": "import Mathlib.Data.Real.Basic"
}
```

或者使用完整代码：
```json
{
  "id": "002",
  "nl_problem": "Prove something",
  "formal_code": "import Mathlib.Data.Real.Basic\ntheorem problem_002 : ... := by sorry"
}
```

**输出** (`data/results.jsonl`):
```json
{
  "id": "001",
  "nl_problem": "What is 2 + 2?",
  "problem_conversion": {
    "performed": true,
    "was_converted": true,
    "original_problem": "What is 2 + 2?",
    "converted_problem": "Prove that 2 + 2 = 4.",
    "explanation": "Converted calculation question to proof statement"
  },
  "formal_code": "theorem ... := by\n  norm_num",
  "formalization_success": true,
  "formalization_attempts": 2,  # 尝试次数
  "formalization_corrections": 1,  # 修复次数（如果使用了修复）
  "proof_success": true,
  "informalization_check": {
    "performed": true,
    "passed": true,
    "similarity_score": 1.0,
    "informalized_text": "Prove that two plus two equals four",
    "explanation": "The statements are semantically equivalent (model judgment: YES)"
  }
}
```

**注**: 
- `problem_conversion` 字段仅在启用问题转换功能时出现
- `informalization_check` 字段仅在启用形式化验证功能时出现

## 命令行参数

```bash
python run_pipeline.py [选项]
```

### 模式选择
- `--mock` - 启用 Mock 模式（测试用，无需模型）
- `--openai` - **已弃用**：启用 OpenAI 模式（云端 API，无需 GPU）。推荐在配置文件中直接配置 OpenAI 模型（使用 `model` 字段）
- （默认）根据配置文件自动选择模式（本地模型或 OpenAI API）

### 文件配置
- `--config PATH` - 配置文件路径（默认: `config/config.yaml`）
- `--input PATH` - 输入 JSONL 文件（覆盖配置文件）
- `--output PATH` - 输出 JSONL 文件（覆盖配置文件）

### 处理范围
- `--start N` - 起始问题索引（默认: 0）
- `--end N` - 结束问题索引（不包含，默认: 处理所有）

### 保存设置
- `--save-interval N` - 每处理 N 个问题保存一次（默认: 1）

### 仅执行部分流程
- `--prove-only` - 仅证明模式：从输入文件读取已有`formal_statement`或`formal_code`，跳过formalization。
  支持批量处理，使用`--start`和`--end`指定范围，或省略以处理所有问题
- `--formalizer-only` - 仅形式化模式：从输入文件读取`nl_problem`，仅进行形式化，跳过证明生成。
  支持批量处理，使用`--start`和`--end`指定范围，或省略以处理所有问题
- **注意**：`--prove-only` 和 `--formalizer-only` 不能同时使用

### 使用示例

```bash
# Mock 模式 - 处理前 10 个问题
python run_pipeline.py --mock --start 0 --end 10

# OpenAI API 模式 - 自定义输入输出（推荐：在配置文件中配置 OpenAI 模型，无需 --openai 参数）
python run_pipeline.py --input data/my_problems.jsonl --output data/my_results.jsonl
# 或使用已弃用的 --openai 参数（向后兼容）
# python run_pipeline.py --openai --input data/my_problems.jsonl --output data/my_results.jsonl

# 本地模式 - 每 5 个问题保存一次
python run_pipeline.py --start 0 --end 100 --save-interval 5

# 断点续传 - 从第 25 个问题继续
python run_pipeline.py --start 25

# 使用自定义配置
python run_pipeline.py --config config/custom_config.yaml --start 0 --end 10

# 证明已有形式化statement（从JSONL文件，支持单个或批量）
# 推荐：在配置文件中配置 OpenAI 模型，无需 --openai 参数
python run_pipeline.py --prove-only --input data/formalized_statements.jsonl --output data/proofs.jsonl

# 证明单个问题（指定索引）
python run_pipeline.py --prove-only --input data/formalized_statements.jsonl --start 0 --end 1

# 仅形式化自然语言问题（从JSONL文件，支持单个或批量）
python run_pipeline.py --formalizer-only --input data/practice_data.jsonl --output data/formalized_only.jsonl

# 仅形式化单个问题（指定索引）
python run_pipeline.py --formalizer-only --input data/practice_data.jsonl --start 0 --end 1

# 注意：--openai 参数已弃用，但仍支持（向后兼容）
```

## 常用命令

```bash
# 测试
python test/test_mock_mode.py

# 测试问题转换功能
python test/example_problem_conversion.py --mock   # Mock 模式（无需 API）
python test/example_problem_conversion.py --api    # 使用真实 API 测试

# 分析结果
python analyze_results.py data/results.jsonl --problem-id 001
```

## 系统要求

**OpenAI 模式**: Python 3.10+, Lean 4, API key  
**本地模式**: Python 3.10+, Lean 4, 16GB+ GPU, 40GB 磁盘

### 多 GPU 配置

如果你有多张 GPU，可以在 `config/config.yaml` 中指定每个模型使用的 GPU 设备：

```yaml
formalizer:
  device: "cuda:0"  # Formalizer 使用 GPU 0

prover:
  device: "cuda:1"  # Prover 使用 GPU 1
```

**可选值**：
- `"cuda:0"`, `"cuda:1"`, `"cuda:2"`, `"cuda:3"` - 指定 GPU 编号
- `"auto"` - 自动分配（默认）

**验证配置**：
```bash
# 查看 GPU 状态
nvidia-smi

# 运行测试
python -m src.pipeline --config config/config.yaml --start 0 --end 1

# 监控 GPU 使用（另开终端）
watch -n 1 nvidia-smi
```

## 故障排除

```bash
# Lean 未安装
lean --version

# API key 未设置
export OPENAI_API_KEY="your-key"

# 清除缓存
rm -rf ~/.cache/lean

# 检查 GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 文档

- [QUICKSTART.md](QUICKSTART.md) - 详细安装指南
- [MOCK_MODE.md](MOCK_MODE.md) - Mock 模式说明
- [ARCHITECTURE.md](ARCHITECTURE.md) - 技术架构
- [PROVE_EXISTING_STATEMENT.md](PROVE_EXISTING_STATEMENT.md) - 使用已有形式化statement直接证明
- [example_problem_conversion.py](test/example_problem_conversion.py) - 问题转换示例
- [example_prove_existing_statement.py](test/example_prove_existing_statement.py) - 直接证明已有statement示例
- [CHANGELOG.md](CHANGELOG.md) - 版本历史

## License

Apache 2.0

---

**Version**: 1.5.0 | [GitHub](https://github.com/your-repo)
