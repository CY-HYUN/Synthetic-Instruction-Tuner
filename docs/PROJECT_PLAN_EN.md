# Synthetic Instruction Tuner

## Project Plan (Detailed Schedule)

---

## 1. Project Overview

### 1.1 Project Information

- **Project Name**: Synthetic Instruction Tuner
- **Duration**: 4 Weeks
- **Budget**: $0 (Free) / $10 (Colab Pro, Optional)
- **Environment**: Google Colab (Free T4 GPU / Colab Pro A100 GPU)

### 1.2 Goal Summary

```
Synthetic Data Generation → Quality Filtering → Preference Data Generation → SFT+DPO Training → Evaluation
```

---

## 2. Overall Schedule Overview

```
Week 1: Environment Setup + Data Generation (Magpie)
Week 2: Quality Filtering + Preference Data Generation
Week 3: Fine-tuning (SFT + DPO)
Week 4: Evaluation + Documentation + Presentation Preparation
```

---

## 3. Week 1: Data Generation

### 3.1 Day 1-2: Environment Setup

#### Task List

| Task                    | Description                         | Estimated Time |
| ----------------------- | ----------------------------------- | -------------- |
| Colab Environment Setup | Check GPU, Install Libraries        | 1 hour         |
| Hugging Face Login      | Token Setup, Model Access Rights    | 30 mins        |
| Project Structure Setup | Folder Structure, Utility Functions | 1 hour         |
| Llama-3.1-8B Test       | Model Loading and Inference Test    | 2 hours        |

#### Checkpoints

- [ ] Confirm T4 GPU allocation in Colab
- [ ] Install transformers, peft, trl
- [ ] Llama-3.1-8B Loading Success (4-bit quantization)

### 3.2 Day 3-5: Magpie Data Generation

#### Task List

| Task                   | Description                   | T4 Time                | A100 Time        |
| ---------------------- | ----------------------------- | ---------------------- | ---------------- |
| Magpie Generator Class | Template-based Generation     | 3 hours                | 3 hours          |
| Instruction Generation | Target 1,500                  | 8-10 hours (Overnight) | 3-4 hours        |
| Response Generation    | Response for each instruction | 8-10 hours (Overnight) | 3-4 hours        |
| Data Storage           | JSON/Parquet Format           | 1 hour                 | 1 hour           |

#### Generation Strategy

**T4 Strategy (12-hour runtime limit)**:
```
Day 3 Night: Start Instruction Generation (Target 500)
Day 4 Morning: Check Checkpoints, Restart
Day 4 Night: Complete Instruction + Start Response Generation
Day 5: Complete Response, Data Verification
```

**A100 Strategy (Single session - unlimited runtime)**:
```
Single Session: Generate all 1,500 samples continuously
Checkpoint interval: 100 samples (optimized for speed)
```

#### Checkpoints

- [ ] MagpieGenerator Class Implementation Complete
- [ ] Instruction 1,500 Generation Complete
- [ ] Response 1,500 Generation Complete
- [ ] Save data/raw/instructions_raw.json

### 3.3 Week 1 Deliverables

| Deliverable         | Path                                    | Description           |
| ------------------- | --------------------------------------- | --------------------- |
| Setup Notebook      | notebooks/01_setup.ipynb                | Environment Setup     |
| Generation Notebook | notebooks/02_magpie_generation.ipynb    | Data Generation       |
| Generator Class     | src/data_generation/magpie_generator.py | Magpie Implementation |
| Raw Data            | data/raw/instructions_raw.json          | 1,500 items           |

---

## 4. Week 2: Filtering + Preference Data

### 4.1 Day 1-2: Quality Filtering

#### Task List

| Task                        | Description              | Estimated Time |
| --------------------------- | ------------------------ | -------------- |
| Filter Class Implementation | 5 Filter Rules           | 3 hours        |
| Execute Filtering           | 1,500 → 1,000            | 1 hour         |
| Statistical Analysis        | Removal Count by Filter  | 1 hour         |
| Result Verification         | Manual Review of Samples | 1 hour         |

#### Filter Rules Details

```python
FILTER_CONFIG = {
    "length": {"min": 20, "max": 500},  # Word count
    "repetition": {"max_repeat": 3},     # Max consecutive repetitions allowed
    "diversity": {"jaccard_threshold": 0.8},
    "refusal_keywords": [
        "I'm an AI", "I cannot", "I don't have",
        "As an AI", "I'm not able"
    ],
    "language": "en"
}
```

#### Checkpoints

- [ ] QualityFilter Class Implementation Complete
- [ ] Filtering Complete (Over 1,000 passed)
- [ ] Generate Statistical Report
- [ ] Save data/filtered/instructions_filtered.json

### 4.2 Day 3-5: Preference Data Generation

#### Task List

| Task                            | Description                 | Estimated Time |
| ------------------------------- | --------------------------- | -------------- |
| Response Generation Model Setup | 3 Small Models              | 2 hours        |
| Multi-Response Generation       | 3 Responses per Instruction | 6-8 hours      |
| Reward Model Setup              | OpenAssistant RM            | 1 hour         |
| Scoring and Selection           | chosen/rejected pairs       | 2-3 hours      |
| Data Verification               | Check Score Distribution    | 1 hour         |

#### Memory Management Strategy

```python
# Sequential Model Loading (Save Memory)
for model_name in ["llama-3.2-1b", "mistral-7b", "qwen2.5-3b"]:
    model = load_model(model_name)
    responses = generate(model, instructions)
    save_responses(responses, model_name)
    del model  # Release Memory
    torch.cuda.empty_cache()
```

#### Checkpoints

- [ ] 3 Models Response Generation Complete
- [ ] Reward Model Scoring Complete
- [ ] Generate 600 Preference Pairs
- [ ] Save data/preference/preference_pairs.json

### 4.3 Week 2 Deliverables

| Deliverable                    | Path                                     | Description                |
| ------------------------------ | ---------------------------------------- | -------------------------- |
| Filtering Notebook             | notebooks/03_quality_filtering.ipynb     | Quality Filter             |
| Preference Generation Notebook | notebooks/04_preference_generation.ipynb | Preference Data            |
| Filter Class                   | src/filtering/quality_filter.py          | Filter Implementation      |
| Preference Builder             | src/preference/preference_builder.py     | Preference Pair Generation |
| Filtered Data                  | data/filtered/instructions_filtered.json | 1,000 items                |
| Preference Data                | data/preference/preference_pairs.json    | 600 items                  |

---

## 5. Week 3: Fine-tuning

### 5.1 Day 1-3: SFT (Supervised Fine-Tuning)

#### Task List

| Task                | Description           | T4 Time    | A100 Time  |
| ------------------- | --------------------- | ---------- | ---------- |
| Dataset Preparation | SFT Format Conversion | 1 hour     | 1 hour     |
| LoRA Setup          | Hyperparameter Tuning | 1 hour     | 1 hour     |
| Llama-3.2-3B SFT    | Execute Training      | 6-8 hours  | 2-4 hours  |
| Mistral-7B SFT      | Execute Training      | 8-10 hours | 3-5 hours  |
| Qwen2.5-3B SFT      | Execute Training      | 6-8 hours  | 2-4 hours  |

#### LoRA Setup

**T4 Configuration (Default)**:
```python
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

TRAINING_ARGS_T4 = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "save_steps": 500,
    "logging_steps": 100
}
```

**A100 Configuration (Optimized)**:
```python
TRAINING_ARGS_A100 = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 12,    # 4 → 12 (3x increase)
    "gradient_accumulation_steps": 2,      # 4 → 2 (reduced)
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "save_steps": 200,                     # 500 → 200 (more frequent)
    "logging_steps": 10
}
# Expected time: T4 6-8h → A100 2-4h (2-3x faster)
```

#### Training Schedule

```
Day 1 Night: Start Llama-3.2-3B SFT
Day 2 Morning: Check Checkpoints
Day 2 Night: Start Mistral-7B SFT (or continue Llama)
Day 3: Qwen2.5-3B SFT + Full Verification
```

#### Checkpoints

- [ ] SFT Dataset Preparation Complete
- [ ] Llama-3.2-3B SFT Complete
- [ ] Mistral-7B SFT Complete
- [ ] Qwen2.5-3B SFT Complete
- [ ] Save 3 checkpoints in models/sft/ folder

### 5.2 Day 4-5: DPO (Direct Preference Optimization)

#### Task List

| Task                    | Description               | T4 Time   | A100 Time |
| ----------------------- | ------------------------- | --------- | --------- |
| DPO Dataset Preparation | Format Conversion         | 1 hour    | 1 hour    |
| Llama-3.2-3B DPO        | Start from SFT Checkpoint | 4-6 hours | 1-2 hours |
| Mistral-7B DPO          | Start from SFT Checkpoint | 5-7 hours | 2-3 hours |
| Qwen2.5-3B DPO          | Start from SFT Checkpoint | 4-6 hours | 1-2 hours |

#### DPO Setup

**T4 Configuration (Default)**:
```python
DPO_CONFIG_T4 = {
    "beta": 0.1,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8
}
```

**A100 Configuration (Optimized)**:
```python
DPO_CONFIG_A100 = {
    "beta": 0.1,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,    # 2 → 8 (4x increase)
    "gradient_accumulation_steps": 2,    # 8 → 2 (reduced)
    "save_steps": 100                     # 200 → 100 (more frequent)
}
# Memory: ~30-35GB (policy + reference model loaded simultaneously)
# Expected time: T4 4-6h → A100 1-2h (3-4x faster)
```

#### Checkpoints

- [ ] DPO Dataset Preparation Complete
- [ ] Llama-3.2-3B DPO Complete
- [ ] Mistral-7B DPO Complete
- [ ] Qwen2.5-3B DPO Complete
- [ ] Save 3 final checkpoints in models/dpo/ folder

### 5.3 Week 3 Deliverables

| Deliverable  | Path                             | Description  |
| ------------ | -------------------------------- | ------------ |
| SFT Notebook | notebooks/05_sft_training.ipynb  | SFT Training |
| DPO Notebook | notebooks/06_dpo_training.ipynb  | DPO Training |
| SFT Models   | models/sft/{llama,mistral,qwen}/ | 3 Models     |
| DPO Models   | models/dpo/{llama,mistral,qwen}/ | 3 Models     |

---

## 6. Week 4: Evaluation + Documentation

### 6.1 Day 1-2: Benchmark Evaluation

#### Task List

| Task                  | Description               | Estimated Time |
| --------------------- | ------------------------- | -------------- |
| lm-eval Setup         | Environment Configuration | 1 hour         |
| Base Model Evaluation | 3 Models × 3 Benchmarks   | 3-4 hours      |
| SFT Model Evaluation  | 3 Models × 3 Benchmarks   | 3-4 hours      |
| DPO Model Evaluation  | 3 Models × 3 Benchmarks   | 3-4 hours      |
| Result Analysis       | Comparison Table, Charts  | 2 hours        |

#### Evaluation Commands

```bash
# IFEval Evaluation
lm_eval --model hf \
    --model_args pretrained=./models/dpo/llama \
    --tasks ifeval \
    --batch_size 4 \
    --output_path ./evaluation/results/llama_dpo_ifeval.json

# MT-Bench, MMLU in the same way
```

#### Checkpoints

- [ ] Base Model 3 Evaluations Complete
- [ ] SFT Model 3 Evaluations Complete
- [ ] DPO Model 3 Evaluations Complete
- [ ] Comparison Table Generation Complete

### 6.2 Day 3: Agent Evaluation (Additional)

#### Task List

| Task                 | Description                       | Estimated Time |
| -------------------- | --------------------------------- | -------------- |
| AgentBench Setup     | Environment Configuration         | 1 hour         |
| DPO Model Evaluation | 3 Models Evaluation               | 2-3 hours      |
| Result Analysis      | Integration with existing results | 1 hour         |

#### Checkpoints

- [ ] AgentBench Setup Complete
- [ ] 3 DPO Models Agent Evaluation Complete
- [ ] Result Integration Complete

### 6.3 Day 4-5: Documentation + Presentation Preparation

#### Task List

| Task                           | Description                       | Estimated Time |
| ------------------------------ | --------------------------------- | -------------- |
| Technical Report Writing       | Explanation of the entire process | 4-6 hours      |
| Presentation Material Creation | PPT/Slides                        | 3-4 hours      |
| Code Cleanup                   | Comments, README                  | 2 hours        |
| Hugging Face Upload            | Models + Datasets                 | 2 hours        |

#### Report Structure

```markdown
1. Introduction - Project Purpose, Background
2. Related Work - Magpie, DPO, LoRA
3. Methodology - Pipeline Description
4. Experimental Setup - Models, Data, Hyperparameters
5. Results - Benchmark Results, Analysis
6. Discussion - Success Factors, Limitations
7. Conclusion
```

#### Checkpoints

- [ ] Technical Report Complete
- [ ] Presentation Materials Complete
- [ ] README.md Writing Complete
- [ ] Hugging Face Hub Upload Complete

### 6.4 Week 4 Deliverables

| Deliverable         | Path                                    | Description     |
| ------------------- | --------------------------------------- | --------------- |
| Evaluation Notebook | notebooks/07_benchmark_evaluation.ipynb | Benchmark       |
| Agent Notebook      | notebooks/08_agent_evaluation.ipynb     | AgentBench      |
| Evaluation Results  | evaluation/results/                     | JSON Files      |
| Comparison Table    | evaluation/comparison_table.md          | Result Summary  |
| Visualization       | evaluation/figures/                     | Chart Images    |
| Technical Report    | docs/TECHNICAL_REPORT.md                | Detailed Report |

---

## 7. Milestones

| Milestone               | Completion Criteria                 | Target Date  |
| ----------------------- | ----------------------------------- | ------------ |
| M1: Environment Prep    | Colab + Model Loading Success       | Week 1 Day 2 |
| M2: Data Generation     | 1,500 raw data items                | Week 1 Day 5 |
| M3: Data Cleaning       | 1,000 filtered + 600 preference     | Week 2 Day 5 |
| M4: SFT Complete        | 3 Model SFT Checkpoints             | Week 3 Day 3 |
| M5: DPO Complete        | 3 Model Final Checkpoints           | Week 3 Day 5 |
| M6: Evaluation Complete | All Benchmark Results               | Week 4 Day 3 |
| M7: Project Complete    | Report + Presentation Materials     | Week 4 Day 5 |

---

## 8. Risk Response Plan

### 8.1 In Case of Time Delay

| Situation             | Response                              |
| --------------------- | ------------------------------------- |
| Data Generation Delay | Reduce to 1,000                       |
| Model Training Delay  | Exclude Mistral-7B (Only 2 3B models) |
| Evaluation Delay      | IFEval Mandatory, Others Optional     |

### 8.2 In Case of Technical Issues

| Situation                 | Response                                               |
| ------------------------- | ------------------------------------------------------ |
| GPU OOM                   | Reduce batch size to 2, increase gradient accumulation |
| Model Convergence Failure | Adjust learning rate, increase epochs                  |
| Colab Disconnection       | Restart from checkpoint                                |

---

## 9. Change History

| Version | Date | Change Content | Author |
| ------- | ---- | -------------- | ------ |
| 1.0     | TBD  | Initial Draft  | -      |

---

_This plan is subject to change depending on the situation during the project._
