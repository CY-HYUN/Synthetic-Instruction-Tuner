# 05b_prompt_tuning.ipynb - Production Checklist

## ✅ COMPLETED on A100 GPU (2025-12-26)

**실제 학습 결과:**
- 학습 시간: 18.8분 (A100)
- Eval Loss: 2.979
- Train Loss: 5.223
- 총 Steps: 114
- 비용: ~1.68 compute units
- Trainable Parameters: 61,440 (0.003% of model)

---

## ✅ All Issues Fixed - Ready for A100 GPU Training

### 🔧 Fixed Issues

#### 1. **SFTTrainer Incompatibility** ✅
- ❌ SFTTrainer → ✅ Standard `Trainer` with `DataCollatorForLanguageModeling`
- **Reason**: SFTTrainer calls `merge_and_unload()` which Prompt Tuning doesn't support
- **Solution**: Cell 13, 22, 26 modified

#### 2. **Tokenization & Padding** ✅
- ❌ Variable sequence lengths → ✅ `padding="max_length"`
- **Error Fixed**: `ValueError: expected sequence of length 430 at dim 1 (got 586)`
- **Solution**: Added padding in Cell 22

#### 3. **Dataset Processing** ✅
- ❌ Dataset already processed → ✅ Reload dataset in Cell 22
- **Error Fixed**: `KeyError: 'instruction'`
- **Solution**: Reload train/val datasets from original data

#### 4. **A100 Optimization** ✅
- **Batch size**: 12 (optimized for A100 40GB VRAM)
- **Gradient accumulation**: 2 (Effective batch: 24)
- **BF16**: True (A100 native support)
- **TF32**: True (additional speedup)

---

## 📊 Training Configuration

### Expected Behavior
```
Training samples: 900
Validation samples: 100
Total steps: ~114 (3 epochs × 900 / 24)

Virtual Tokens: 20
Init Method: RANDOM

Total training time: 18.8 minutes on A100
Peak Memory: ~5.94 GB
```

### Comparison with LoRA
| Metric | LoRA | Prompt Tuning | Winner |
|--------|------|---------------|--------|
| Trainable Params | 12.16M | 61K | 🏆 PT (197x fewer) |
| Training Time | 8.2 min | 18.8 min | 🏆 LoRA (2.3x faster) |
| Eval Loss | 0.541 | 2.979 | 🏆 LoRA (5.5x better) |
| Model Size | ~50 MB | ~1 MB | 🏆 PT (50x smaller) |
| Inference Speed | 7.70 tok/s | 8.44 tok/s | 🏆 PT (9.6% faster) |

---

## 🚀 Cell-by-Cell Modifications

### Cell 13: Update Imports
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,  # ✅ Added
    DataCollatorForLanguageModeling  # ✅ Added
)
```

### Cell 22: Tokenization Function
```python
def format_and_tokenize(sample):
    """Format and tokenize instruction-response pair for training."""
    instruction = sample["instruction"]
    response = sample["output"]

    # Same format as LoRA for fair comparison
    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"

    # Tokenize with padding to max_length
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=config['training']['sft_max_seq_length'],
        padding="max_length",  # ✅ Added padding
        return_tensors=None,
    )

    # Add labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

# Reload datasets to ensure we have original columns
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Apply formatting and tokenization
train_dataset = train_dataset.map(
    format_and_tokenize,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train dataset"
)
val_dataset = val_dataset.map(
    format_and_tokenize,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing val dataset"
)
```

### Cell 26: Trainer Replacement
```python
# Create data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# Use standard Trainer (not SFTTrainer) for Prompt Tuning
# SFTTrainer tries to call merge_and_unload() which Prompt Tuning doesn't support
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,  # ✅ Added
)

print("Trainer initialized!")
print("Note: Using standard Trainer (not SFTTrainer) for Prompt Tuning compatibility")
```

---

## 🔴 Error Scenarios & Solutions

### Error 1: AttributeError - merge_and_unload
```
AttributeError: 'PeftModelForCausalLM' object has no attribute 'merge_and_unload'
```
**Solution**: Use standard `Trainer` instead of `SFTTrainer` (Cell 26)

### Error 2: ValueError - Sequence Length Mismatch
```
ValueError: expected sequence of length 430 at dim 1 (got 586)
```
**Solution**: Add `padding="max_length"` in tokenizer (Cell 22)

### Error 3: KeyError - Missing Column
```
KeyError: 'instruction'
```
**Solution**: Reload dataset from original data (Cell 22)

---

## 📁 Output Files

After successful completion:

```
/content/drive/MyDrive/synthetic-instruction-tuner/
├── models/prompt_tuning/
│   ├── checkpoint/
│   │   ├── checkpoint-100/
│   │   ├── checkpoint-114/
│   │   └── logs/
│   └── final/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors  (~1 MB)
│       ├── training_config.json
│       └── tokenizer files
├── evaluation/
│   └── metrics/
│       └── prompt_tuning_metrics.json
```

---

## ✅ Success Criteria

Training is successful if:
1. ✅ Cell 27 (Training) runs without errors
2. ✅ Loss decreases over time (logged every 10 steps)
3. ✅ Checkpoints saved at steps 100, 114
4. ✅ Final model saved to `models/prompt_tuning/final/`
5. ✅ Metrics JSON saved with correct values
6. ✅ Total time: ~18-20 minutes on A100

---

## 📊 Key Findings

### When to Use Prompt Tuning
✅ **Best for:**
- Multi-tenant serving (many adapters per model)
- Edge deployment (mobile, IoT devices)
- Resource-constrained environments
- Rapid prototyping and A/B testing
- Scenarios where model size matters

❌ **Not ideal for:**
- Production apps requiring highest quality
- Single-tenant deployments
- When training time is critical
- Applications sensitive to quality degradation

### When to Use LoRA Instead
✅ **Best for:**
- Production applications prioritizing quality
- Single-tenant deployments
- Domain-specific fine-tuning (legal, medical)
- When 50MB adapter size is acceptable
- Applications where quality is critical

---

**Status**: ✅ COMPLETED on A100 GPU (2025-12-26)
**Training Time**: 18.8 minutes
**Cost**: ~1.68 compute units
**Eval Loss**: 2.979 (vs LoRA 0.541)
