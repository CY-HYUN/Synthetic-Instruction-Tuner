# Synthetic Instruction Tuner

## Technology Stack and Environment Setup Guide

---

## 1. Development Environment

### 1.1 Primary Environment: Google Colab

**Colab Free (T4 GPU)**
```
GPU: NVIDIA T4 (16GB VRAM) - Free
RAM: 12.7GB (Free)
Storage: Google Drive Integration
Runtime: Max 12 hours (Limited)
```

**Colab Pro (A100 GPU) - Optimized ⚡**
```
GPU: NVIDIA A100 (40GB VRAM) - Pro/Pro+
RAM: 25-52GB
Storage: Google Drive Integration
Runtime: Unlimited (Pro+) / 24 hours (Pro)
```

**This project is optimized for Colab Pro A100:**
- 3-4x larger batch sizes (SFT: 12, DPO: 8)
- Optimized checkpoint intervals (100 samples)
- 2-3x faster overall pipeline
- Complete dataset generation in single session

### 1.2 Colab Setup Method

```python
# Check GPU
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set Project Path
PROJECT_PATH = "/content/drive/MyDrive/synthetic-instruction-tuner"
```

---

## 2. Python Packages

### 2.1 Core Libraries

```bash
# Colab Installation Script
!pip install -q transformers==4.36.0
!pip install -q peft==0.7.0
!pip install -q trl==0.7.4
!pip install -q datasets==2.16.0
!pip install -q accelerate==0.25.0
!pip install -q bitsandbytes==0.41.3
!pip install -q lm-eval==0.4.0
!pip install -q sentencepiece
!pip install -q protobuf
```

### 2.2 Package Description

| Package       | Version | Purpose                                              |
| ------------- | ------- | ---------------------------------------------------- |
| transformers  | 4.36+   | Model Loading, Tokenizer, Inference                  |
| peft          | 0.7+    | LoRA Implementation, Parameter Efficient Fine-tuning |
| trl           | 0.7+    | SFTTrainer, DPOTrainer                               |
| datasets      | 2.16+   | Dataset Management, Preprocessing                    |
| accelerate    | 0.25+   | Distributed Training, Memory Optimization            |
| bitsandbytes  | 0.41+   | 4-bit/8-bit Quantization                             |
| lm-eval       | 0.4+    | Benchmark Evaluation (IFEval, MMLU, etc.)            |
| sentencepiece | -       | Llama Tokenizer Dependency                           |

---

## 3. Model Information

### 3.1 Data Generation Model

```python
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Features
# - Parameters: 8B
# - Context Length: 128K
# - License: Llama 3.1 Community License
# - Purpose: Magpie style instruction generation

# Loading Code
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 3.2 Fine-tuning Target Models (3 Models)

#### Llama-3.2-3B

```python
MODEL_ID = "meta-llama/Llama-3.2-3B"
# - Parameters: 3B
# - VRAM Requirement: ~6GB (4-bit)
# - Features: Latest Llama Architecture, Efficient
# - Access: Requires Llama 3.2 License Agreement
```

#### Mistral-7B-v0.1

```python
MODEL_ID = "mistralai/Mistral-7B-v0.1"
# - Parameters: 7B
# - VRAM Requirement: ~8GB (4-bit)
# - Features: Sliding Window Attention, High Performance
# - Access: Public Model (No License Agreement Required)
```

#### Qwen2.5-3B

```python
MODEL_ID = "Qwen/Qwen2.5-3B"
# - Parameters: 3B
# - VRAM Requirement: ~6GB (4-bit)
# - Features: Multilingual Support, Developed by Alibaba China
# - Access: Public Model
```

### 3.3 Reward Model

```python
MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"

# Loading Code
from transformers import pipeline

reward_model = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=0  # GPU
)

# Usage Example
score = reward_model("User: What is AI?\nAssistant: AI is...")
print(score[0]['score'])  # 0.82
```

---

## 4. Hugging Face Setup

### 4.1 Token Generation

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `synthetic-instruction-tuner`
4. Role: `write` (For model upload)
5. Copy token and save securely

### 4.2 Login from Colab

```python
from huggingface_hub import login

# Method 1: Direct Input
login(token="hf_xxxxxxxxxxxxx")

# Method 2: Environment Variable (Recommended)
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxx"
login()
```

### 4.3 Model Access Rights

**Models requiring license agreement:**

- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-3B

**Procedure:**

1. Visit each model page
2. Click "Access repository" button
3. Enter information and agree to license
4. Wait for approval (Usually instant to a few hours)

---

## 5. LoRA Configuration

### 5.1 LoRA Configuration

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,                           # Rank: Lower saves memory
    lora_alpha=16,                 # Scaling factor
    target_modules=[               # Target layers
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

### 5.2 Memory Usage Comparison

| Setting          | 3B Model | 7B Model |
| ---------------- | -------- | -------- |
| Full Fine-tuning | ~24GB    | ~56GB    |
| LoRA (r=8)       | ~6GB     | ~10GB    |
| LoRA (r=4)       | ~5GB     | ~8GB     |

### 5.3 Recommended Settings (Colab T4 16GB)

```python
# 3B Model
LORA_R = 8
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

# 7B Model
LORA_R = 8
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
```

---

## 6. Training Configuration

### 6.1 SFT Training Arguments

```python
from transformers import TrainingArguments

sft_training_args = TrainingArguments(
    output_dir="./models/sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none"
)
```

### 6.2 DPO Training Arguments

```python
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir="./models/dpo",
    beta=0.1,
    learning_rate=5e-5,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=50,
    save_steps=200,
    report_to="none"
)
```

---

## 7. Evaluation Configuration

### 7.1 lm-evaluation-harness Setup

```bash
# Install
pip install lm-eval

# Basic Execution
lm_eval --model hf \
    --model_args pretrained=./models/dpo/llama,trust_remote_code=True \
    --tasks ifeval,mmlu \
    --batch_size 4 \
    --output_path ./evaluation/results/
```

### 7.2 Evaluation Tasks

| Task       | Description            | Metric   |
| ---------- | ---------------------- | -------- |
| ifeval     | Instruction Following  | Accuracy |
| mmlu       | Knowledge Evaluation   | Accuracy |
| truthfulqa | Truthfulness           | MC1, MC2 |
| hellaswag  | Common Sense Reasoning | Accuracy |

### 7.3 AgentBench Setup

```bash
# Clone AgentBench
git clone https://github.com/THUDM/AgentBench.git

# Simple Evaluation (webshop)
cd AgentBench
python eval.py --model ./models/dpo/llama --task webshop
```

---

## 8. Troubleshooting

### 8.1 GPU OOM (Out of Memory)

```python
# Solution 1: Reduce Batch Size
per_device_train_batch_size = 2  # 4 → 2
gradient_accumulation_steps = 8  # 4 → 8

# Solution 2: Stronger Quantization
load_in_4bit = True
bnb_4bit_compute_dtype = torch.float16

# Solution 3: Clear Memory
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### 8.2 Colab Disconnection

```python
# Restart from Checkpoint
trainer.train(resume_from_checkpoint="./checkpoint-latest")

# Auto Checkpoint
save_steps = 500  # Save every 500 steps
```

### 8.3 Model Loading Failure

```python
# Check Token
from huggingface_hub import whoami
print(whoami())  # Check login status

# Check License
# Visit https://huggingface.co/meta-llama to check agreement
```

---

## 9. Checklist

### 9.1 Check Before Project Start

- [ ] Prepare Google Account
- [ ] Create Hugging Face Account
- [ ] Generate Hugging Face Token (write permission)
- [ ] Agree to Llama 3.1, 3.2 Licenses
- [ ] Check Google Drive Free Space (Min 20GB)

### 9.2 Check at Week 1 Start

- [ ] Check Colab GPU Allocation (Runtime > Change runtime type > T4)
- [ ] Complete All Package Installations
- [ ] Hugging Face Login Success
- [ ] Test Model Loading Success

### 9.3 Periodic Checks

- [ ] Check Checkpoint Saving
- [ ] Check Google Drive Sync
- [ ] Monitor GPU Memory Usage
- [ ] Check Training Logs

---

_This document may be updated during the project._
