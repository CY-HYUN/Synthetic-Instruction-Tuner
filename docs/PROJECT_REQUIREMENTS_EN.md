# Synthetic Instruction Tuner

## Project Requirements Specification

---

## 1. Project Overview

### 1.1 Project Name

**Synthetic Instruction Tuner**

> End-to-End Synthetic Data Generation and Fine-tuning Pipeline for Instruction-Tuned LLMs

### 1.2 Project Objectives

1. **Academic Goal**: Fulfill final project requirements for the LLM course
2. **Technical Goal**: Build an End-to-End pipeline from synthetic data generation to fine-tuning and evaluation

### 1.3 Project Duration

- **Estimated Duration**: 4 Weeks
- **Start Date**: TBD
- **End Date**: TBD

### 1.4 Budget

- **Target Budget**: $0 (Completely Free)
- **Max Allowable Budget**: $10 (Colab Pro 1 month, Optional)

---

## 2. Project Background

### 2.1 Professor's Requirements (Based on Lecture Content)

Key requirements extracted from lecture transcripts:

| Item                      | Requirement                                                     | Importance |
| ------------------------- | --------------------------------------------------------------- | ---------- |
| NLP Task                  | Free Choice                                                     | Mandatory  |
| LLM Usage                 | Implementation with step-by-step understanding                  | Mandatory  |
| Fine-tuning               | Apply fine-tuning techniques                                    | Mandatory  |
| Evaluation                | Performance evaluation and comparison                           | Mandatory  |
| Code Quality              | Clean structure, reproducibility                                | Mandatory  |
| Report                    | Include process explanation                                     | Mandatory  |
| Presentation              | Presentation materials                                          | Mandatory  |
| **Process Understanding** | **Understanding the process is more important than the result** | **Core**   |

---

## 3. Technical Requirements

### 3.1 Technology Stack

#### 3.1.1 Programming Language

- **Python 3.10+**

#### 3.1.2 Core Libraries

| Library      | Purpose                 | Version |
| ------------ | ----------------------- | ------- |
| transformers | Model Loading/Inference | 4.36+   |
| peft         | LoRA Fine-tuning        | 0.7+    |
| trl          | SFT/DPO Trainer         | 0.7+    |
| datasets     | Dataset Management      | 2.16+   |
| lm-eval      | Benchmark Evaluation    | 0.4+    |
| torch        | Deep Learning Framework | 2.1+    |
| bitsandbytes | Quantization            | 0.41+   |
| accelerate   | Distributed Training    | 0.25+   |

#### 3.1.3 Development Environment

- **Primary**: Google Colab (T4 GPU, Free)
- **Backup**: Colab Pro (A100/V100, $10/month)
- **Local**: VSCode + Python

### 3.2 Model Requirements

#### 3.2.1 Data Generation Model

| Model                            | Purpose                | Size |
| -------------------------------- | ---------------------- | ---- |
| meta-llama/Llama-3.1-8B-Instruct | Magpie Data Generation | 8B   |

#### 3.2.2 Fine-tuning Target Models (3 Models)

| Model                     | Size | Reason for Selection                    |
| ------------------------- | ---- | --------------------------------------- |
| meta-llama/Llama-3.2-3B   | 3B   | Latest Llama, Efficient                 |
| mistralai/Mistral-7B-v0.1 | 7B   | Excellent Performance, Active Community |
| Qwen/Qwen2.5-3B           | 3B   | Multilingual Support, Diversity         |

#### 3.2.3 Evaluation/Auxiliary Models

| Model                                          | Purpose                 | Cost |
| ---------------------------------------------- | ----------------------- | ---- |
| OpenAssistant/reward-model-deberta-v3-large-v2 | Preference Data Scoring | $0   |

### 3.3 Data Requirements

#### 3.3.1 Generated Data Scale

| Data Type             | Target Quantity | Purpose            |
| --------------------- | --------------- | ------------------ |
| Raw Instructions      | 1,500 items     | Initial Generation |
| Filtered Instructions | 1,000 items     | After Filtering    |
| Preference Pairs      | 600 items       | For DPO Training   |

#### 3.3.2 Data Quality Standards

- **Length**: 20~500 words
- **Diversity**: Jaccard similarity < 0.8
- **Quality**: No repetitive patterns, no refusal responses

---

## 4. Functional Requirements

### 4.1 Phase 1: Synthetic Data Generation

#### 4.1.1 Magpie Method Implementation

```
Input: Only template prompt provided
      "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"

Process: Llama-3.1-8B-Instruct automatically generates instruction
         Then generates response as well

Output: 1,500 {instruction, response} pairs
```

#### 4.1.2 Implementation Requirements

- [ ] Llama-3.1-8B-Instruct Model Loading (4-bit quantization)
- [ ] Template-based instruction generation function
- [ ] Response generation function
- [ ] Batch processing (Memory efficiency)
- [ ] Checkpoint saving (Every 1,000 items)
- [ ] Progress display

### 4.2 Phase 2: Quality Filtering

#### 4.2.1 Rule-based Filtering

```
Input: 1,500 raw instruction-response pairs

Filter:
1. Length Verification (20-500 words)
2. Repetitive Pattern Detection
3. Diversity Check (Jaccard < 0.8)
4. Refusal Response Filter ("I'm an AI", "I cannot", etc.)
5. Language Consistency (English only)

Output: 1,000 High-Quality Data
```

#### 4.2.2 Implementation Requirements

- [ ] Length filter function
- [ ] Repetitive pattern detection function (Regex)
- [ ] Jaccard similarity calculation function
- [ ] Keyword blacklist filter
- [ ] Statistical report generation (Removal count by filter)

### 4.3 Phase 3: Preference Data Generation

#### 4.3.1 Multi-Model Response Generation

```
Input: 1,000 instructions

Process:
1. 3 models generate responses for each instruction
2. Score with OpenAssistant Reward Model
3. Select highest score vs lowest score

Output: 600 {instruction, chosen, rejected} pairs
```

#### 4.3.2 Implementation Requirements

- [ ] 3 Model Loading Function (Sequential, Memory Management)
- [ ] Response Generation Function
- [ ] Reward Model Scoring Function
- [ ] Preference Pair Generation Function
- [ ] Score Distribution Analysis

### 4.4 Phase 4: Fine-tuning

#### 4.4.1 SFT (Supervised Fine-Tuning)

```
Input: 1,000 instruction-response pairs
Model: Each of the 3 base models

Settings:
- LoRA: r=8, alpha=16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Epochs: 3
- Batch size: 4 (gradient accumulation: 4)
- Learning rate: 2e-4

Output: 3 SFT Model Checkpoints
```

#### 4.4.2 DPO (Direct Preference Optimization)

```
Input: 600 preference pairs
Model: 3 SFT Checkpoints

Settings:
- Beta: 0.1
- Learning rate: 5e-5
- Epochs: 1

Output: 3 SFT+DPO Model Checkpoints
```

#### 4.4.3 Implementation Requirements

- [ ] LoRA Setup Function
- [ ] SFTTrainer Setup and Training
- [ ] DPOTrainer Setup and Training
- [ ] Checkpoint Save/Load
- [ ] Training Log Recording (wandb optional)
- [ ] OOM Prevention Settings

### 4.5 Phase 5: Evaluation

#### 4.5.1 Standard Benchmark Evaluation

```
Evaluation Targets:
- 3 Base Models (Comparison Baseline)
- 3 SFT Models
- 3 SFT+DPO Models

Benchmarks:
- IFEval: Instruction Following Ability
- MT-Bench: Conversation Quality
- MMLU: Knowledge Understanding

Tool: lm-evaluation-harness
```

#### 4.5.2 Agent Evaluation (Additional)

```
Evaluation Targets: 3 Final SFT+DPO Models

Benchmarks:
- AgentBench (webshop subset)

Purpose: Verify Agentic LLM capabilities (Internship Appeal)
```

#### 4.5.3 Implementation Requirements

- [ ] lm-eval Setup Script
- [ ] Result Parsing Function
- [ ] Comparison Table Generation
- [ ] Visualization (Charts)
- [ ] AgentBench Setup

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

- Complete entire pipeline within 4 weeks
- Executable on Colab Free version
- Save checkpoints within 12-hour limit

### 5.2 Maintainability

- Modularized code structure
- Clear function/class documentation
- Reproducible experiments (Fixed seed)

### 5.3 Scalability

- Easy to add new models
- Filter rules can be added/modified
- Evaluation benchmarks can be extended

---

## 6. Constraints

### 6.1 Technical Constraints

| Constraint       | Description            | Response Plan            |
| ---------------- | ---------------------- | ------------------------ |
| Colab GPU Memory | T4: 16GB               | 4-bit quantization, LoRA |
| Colab Time Limit | 12 hours               | Save Checkpoints         |
| Model Size       | 7B or less recommended | Focus on 3B models       |

### 6.2 Cost Constraints

| Item           | Limit      | Note                 |
| -------------- | ---------- | -------------------- |
| API Cost       | $0         | API usage prohibited |
| Colab          | Free First | Pro is last resort   |
| Other Services | $0         | Open source only     |

### 6.3 Time Constraints

- Complete within 4 weeks
- Includes presentation material preparation time

---

## 7. Deliverables

### 7.1 Code Deliverables

```
synthetic-instruction-tuner/
├── src/
│   ├── data_generation/      # Magpie Implementation
│   ├── filtering/            # Quality Filtering
│   ├── preference/           # Preference Data Generation
│   ├── training/             # SFT, DPO Training
│   └── evaluation/           # Evaluation Scripts
├── notebooks/                # Colab Notebooks
├── data/                     # Generated Data
├── models/                   # Checkpoints
└── evaluation/               # Evaluation Results
```

### 7.2 Documentation Deliverables

| Document                   | Description         | Format   |
| -------------------------- | ------------------- | -------- |
| Requirements Specification | This Document       | Markdown |
| Project Plan               | Detailed Schedule   | Markdown |
| Technical Report           | Process and Results | PDF      |
| Presentation Materials     | Presentation        | PPT/PDF  |

### 7.3 Model Deliverables

| Model                | Description          | Public           |
| -------------------- | -------------------- | ---------------- |
| Llama-3.2-3B-SFT-DPO | Fine-tuning Complete | Hugging Face Hub |
| Mistral-7B-SFT-DPO   | Fine-tuning Complete | Hugging Face Hub |
| Qwen2.5-3B-SFT-DPO   | Fine-tuning Complete | Hugging Face Hub |

### 7.4 Data Deliverables

| Dataset                | Scale        | Public           |
| ---------------------- | ------------ | ---------------- |
| Synthetic Instructions | 1,000 items | Hugging Face Hub |
| Preference Pairs       | 600 items   | Hugging Face Hub |

---

## 8. Success Criteria

### 8.1 Mandatory Success Criteria

- [ ] Generate over 1,000 synthetic data items
- [ ] Complete SFT+DPO for 3 models
- [ ] Complete IFEval, MT-Bench, MMLU evaluations
- [ ] Confirm performance improvement over Base
- [ ] Complete report and presentation materials

### 8.2 Additional Success Criteria (Internship Appeal)

- [ ] Complete AgentBench evaluation
- [ ] Public on Hugging Face Hub
- [ ] Public GitHub Code
- [ ] Provide reproducible notebooks

### 8.3 Performance Goals (Reference)

| Benchmark | Base | Goal (SFT+DPO)  |
| --------- | ---- | --------------- |
| IFEval    | ~30% | 40%+            |
| MT-Bench  | ~4.0 | 5.0+            |
| MMLU      | ~45% | 45%+ (Maintain) |

---

## 9. Risk Management

### 9.1 Identified Risks

| Risk                      | Impact | Probability | Response Plan                   |
| ------------------------- | ------ | ----------- | ------------------------------- |
| Colab Disconnection       | High   | High        | Save checkpoints frequently     |
| GPU OOM                   | High   | Medium      | Reduce batch size, Quantization |
| Model Convergence Failure | High   | Low         | Adjust hyperparameters          |
| Data Quality Degradation  | Medium | Medium      | Strengthen filter criteria      |
| Time Shortage             | High   | Medium      | Reduce number of models (3→2)   |

### 9.2 Contingency Plan

- **In case of Time Shortage**: Exclude Mistral-7B, proceed with only 2 3B models
- **In case of GPU Shortage**: Pay for Colab Pro 1 month ($10)
- **In case of Low Performance**: Increase data amount or epochs

---

## 10. Change History

| Version | Date | Change Content | Author |
| ------- | ---- | -------------- | ------ |
| 1.0     | TBD  | Initial Draft  | -      |

---

_This document is subject to change during the project._
