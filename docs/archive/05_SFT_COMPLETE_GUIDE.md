# 05_sft_training.ipynb - Complete Guide

## ✅ COMPLETED on A100 GPU (2025-12-26)

**실제 학습 결과:**
- **학습 시간**: 8.2분 (A100)
- **Eval Loss**: 0.541
- **Train Loss**: 0.748
- **총 Steps**: 114
- **비용**: ~0.73 compute units (예상 10-21 대비 **15배 절감**)
- **Trainable Parameters**: 12,156,928 (0.67% of model)

---

## 📋 목차

1. [개요](#개요)
2. [A100 최적화 설정](#a100-최적화-설정)
3. [Fixed Issues](#fixed-issues)
4. [실행 가이드](#실행-가이드)
5. [성공 기준](#성공-기준)
6. [에러 해결](#에러-해결)
7. [출력 파일](#출력-파일)
8. [다음 단계](#다음-단계)

---

## 개요

이 노트북은 **LoRA (Low-Rank Adaptation)**를 사용하여 Llama-3.2-3B 모델을 미세조정합니다.

### 주요 특징
- **Parameter-Efficient**: 전체 파라미터의 0.67%만 학습 (12M / 1.8B)
- **A100 최적화**: BF16 + TF32 지원으로 2-3배 빠른 학습
- **안정적 학습**: 체크포인트 시스템으로 중단 복구 가능
- **비용 효율**: 예상 대비 15배 저렴 (0.73 units vs 10-21 units)

---

## A100 최적화 설정

### 🚀 주요 변경 사항

#### 1. **Batch Size 증가**
- **T4**: 4 → **A100**: 12 (3배 증가)
- **Gradient Accumulation**: 4 → 2 (절반으로)
- **Effective Batch Size**: 16 → 24 (50% 증가)

#### 2. **BF16 활성화**
- **FP16**: False (T4 전용)
- **BF16**: True (A100 네이티브 지원)
- **TF32**: True (추가 성능 향상)

#### 3. **Checkpoint 간격 조정**
- **T4**: 50 steps → **A100**: 100 steps
- 더 빠른 학습으로 체크포인트 빈도 감소
- Total steps: ~114 (900 samples × 3 epochs ÷ 24)

#### 4. **Quantization dtype 변경**
- **T4**: `torch.float16`
- **A100**: `torch.bfloat16`

### 📊 성능 비교

| 항목 | T4 (Free) | A100 (Pro) | 실제 A100 결과 | 개선 |
|-----|-----------|-----------|---------------|------|
| **Batch Size** | 4 | 12 | 12 | 3x |
| **Effective Batch** | 16 | 24 | 24 | 1.5x |
| **학습 시간** | 6-10시간 | 2-4시간 | **8.2분** | **40-70배 빠름!** |
| **Precision** | FP16 | BF16 | BF16 | 더 안정적 |
| **총 Steps** | ~168 | ~112 | 114 | - |
| **비용** | Free | 10-21 units | **0.73 units** | **15배 절감** |
| **Eval Loss** | - | - | **0.541** | 우수 |

### 💰 실제 비용 분석

#### **SFT Training (Notebook 05)**
- **예상**: 10-21 units
- **실제**: **0.73 units** (8.2분 × 5.37 units/hr ÷ 60)
- **절감**: **93-96%**

#### **전체 파이프라인 실제 비용**
- **SFT (05)**: 0.73 units
- **Prompt Tuning (05b)**: 1.68 units
- **총합 (05+05b)**: **2.41 units** (예상 31-58 대비 **20배 절감**)

---

## Fixed Issues

### 🔧 API Compatibility ✅

#### 1. **TRL API Updates**
- ❌ `evaluation_strategy` → ✅ `eval_strategy`
- ❌ `dataset_text_field` → ✅ `formatting_func`
- ❌ `max_seq_length` → ✅ Removed (handled by trainer)
- ❌ `tokenizer` parameter → ✅ Removed from SFTTrainer
- ❌ `packing` parameter → ✅ Removed (not supported)

#### 2. **GPU Optimization** ✅
- **Batch size**: 12 (A100 40GB VRAM)
- **Gradient accumulation**: 2
- **FP16**: False
- **BF16**: True (A100 native support)
- **TF32**: True
- **Checkpoint interval**: 100 steps

#### 3. **Documentation Updates** ✅
- Cell 0: A100 최적화 설명 추가
- Cell 25: 최적화 상세 설명
- Cell 4: Config 자동 적용
- Cell 7: BF16/TF32 활성화

---

## 실행 가이드

### 🎯 A100 실행 방법

#### **1. 런타임 변경**
```
런타임 → 런타임 유형 변경 → GPU 유형: A100
```

#### **2. 세션 시작**
```
런타임 → 런타임 다시 시작
```

#### **3. Cell 순서대로 실행**

**Cell 1-2**: Mount Drive
```
✅ Expected: "Mounted at /content/drive"
```

**Cell 3**: Load config
```
✅ Expected: "Configuration loaded!"
```

**Cell 4**: Verify config (A100 settings auto-applied)
```
✅ Expected:
A100 GPU settings applied!
  Base model: meta-llama/Llama-3.2-3B
  Batch size: 12
  Gradient accumulation: 2
  Effective batch size: 24
  Expected A100 training time: 2-4 hours
```

**Cell 5**: Install libraries
```
✅ Expected: "Libraries installed successfully!"
```

**Cell 6**: Check GPU
```
✅ Expected:
GPU: NVIDIA A100-SXM4-40GB
GPU Memory: 40.00 GB
```

**Cell 7**: BF16/TF32 Setup
```
✅ Expected:
==================================================
A100 GPU Performance Settings:
  BF16 enabled: True
  TF32 enabled: True
  Optimal for A100 40GB VRAM
==================================================
```

**Cell 27**: Training arguments
```
✅ Expected:
Training arguments (A100 optimized):
  Epochs: 3
  Batch size: 12
  Gradient accumulation: 2
  Effective batch size: 24
  Learning rate: 0.0002
  BF16: True, TF32: True
  Total steps: ~112
  Save/eval every: 100 steps
  Expected training time: 2-4 hours on A100
```

**Cell 31**: START TRAINING
```
✅ Expected:
Starting SFT training...
Start time: 2025-12-26 XX:XX:XX
==================================================
{'loss': 1.XXXX, 'learning_rate': 0.000XX, 'epoch': 0.XX}
...
```

### 📊 Training Configuration

**Expected Behavior:**
```
Training samples: 900
Validation samples: 100
Total steps: ~114 (3 epochs × 900 / 24)

Checkpoints saved at:
  - Step 100 (~88%, ~7분)
  - Step 114 (100%, final)

Actual training time: 8.2 minutes on A100 ✅
```

**Memory Usage:**
```
Model loading: ~2.2 GB
After LoRA: ~3-4 GB
During training: ~8-10 GB peak
Safe margin: 30+ GB free (on 40GB A100)
```

---

## 성공 기준

### ✅ 학습 성공 조건

1. ✅ GPU 확인: `GPU: NVIDIA A100-SXM4-40GB`
2. ✅ BF16 활성화: `BF16: True`
3. ✅ Batch size: 12
4. ✅ Effective batch: 24
5. ✅ Cell 31 실행 시 에러 없음
6. ✅ Loss가 시간에 따라 감소
7. ✅ Checkpoint 저장 확인 (step 100, 114)
8. ✅ Final model saved to `models/sft/final/`
9. ✅ Training curves plotted (Cell 42)
10. ✅ **실제 완료 시간: 8.2분** (예상 2-4시간 대비 훨씬 빠름!)

### 📈 실제 학습 결과

```json
{
  "train_loss": 0.748,
  "eval_loss": 0.541,
  "total_steps": 114,
  "training_time_minutes": 8.2,
  "trainable_params": 12156928,
  "trainable_ratio_percent": 0.67,
  "peak_memory_gb": 5.31,
  "inference_speed_tok_per_sec": 7.70
}
```

---

## 에러 해결

### ⚠️ Known Warnings (Safe to Ignore)

```python
UserWarning: Merge lora module to 4-bit linear may get different generations due to rounding errors.
```
- **Reason**: Normal behavior with 4-bit quantization
- **Impact**: None on training quality

```python
UserWarning: Already found a `peft_config` attribute in the model.
```
- **Reason**: LoRA already applied in Cell 20
- **Impact**: None (expected behavior)

### 🔴 Error Scenarios

#### Error 1: `NotImplementedError: BFloat16`
**Solution**: Make sure Cell 27 has `bf16=True` (A100) or `bf16=False` (T4)
```python
# A100
fp16=False,
bf16=True,   # A100 supports BF16
tf32=True,

# T4
fp16=True,
bf16=False,  # T4 doesn't support BF16
```

#### Error 2: `CUDA out of memory`
**Solution**: Reduce batch size
```python
# In Cell 4 or config.json
config['training']['sft_batch_size'] = 8  # Reduce from 12 to 8
```

#### Error 3: `TypeError: unexpected keyword argument`
**Solution**: This notebook is already fixed for latest TRL API
- If still occurs, check TRL version: `!pip show trl`
- Should be `>=0.7.4`

#### Error 4: Wrong GPU Type
**Solution**:
1. 런타임 → 런타임 유형 변경 → A100 선택
2. 런타임 다시 시작
3. Cell 6에서 GPU 확인

---

## 출력 파일

### 📁 학습 완료 후 생성되는 파일

```
/content/drive/MyDrive/synthetic-instruction-tuner/
├── models/sft/
│   ├── sft-checkpoint/
│   │   ├── checkpoint-100/              # ~88% progress
│   │   ├── checkpoint-114/              # Final checkpoint
│   │   └── logs/
│   └── final/
│       ├── adapter_config.json          # LoRA configuration
│       ├── adapter_model.safetensors    # ~50 MB (LoRA weights)
│       ├── training_config.json         # Training metadata ✅
│       └── tokenizer files
├── evaluation/
│   ├── figures/
│   │   └── sft_training_curves.png      # Loss curves ✅
│   └── metrics/
│       └── lora_metrics.json            # Efficiency metrics ✅
```

### 📊 training_config.json 내용

```json
{
  "base_model": "meta-llama/Llama-3.2-3B",
  "training_data_size": 900,
  "validation_data_size": 100,
  "lora_config": {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": [
      "down_proj", "q_proj", "gate_proj", "up_proj",
      "v_proj", "k_proj", "o_proj"
    ]
  },
  "training_args": {
    "epochs": 3,
    "batch_size": 12,
    "gradient_accumulation_steps": 2,
    "learning_rate": 0.0002
  },
  "results": {
    "train_loss": 0.748087699998889,
    "eval_loss": 0.540707528591156,
    "total_steps": 114
  },
  "timestamp": "2025-12-26 13:44:13"
}
```

---

## 다음 단계

### 🎯 SFT 완료 후

#### **Option 1: Prompt Tuning 비교** (권장)
```
✅ Notebook 05b: Prompt Tuning (A100, 18.8분, 1.68 units)
→ LoRA vs Prompt Tuning 비교 분석 가능
```

#### **Option 2: DPO Training**
```
✅ Notebook 06: DPO Training (A100, 예상 1-2시간)
→ Preference alignment 적용
```

#### **Option 3: Evaluation**
```
✅ Notebook 07: Benchmark Evaluation
✅ Notebook 08: Agent Evaluation
✅ Notebook 09: Comparative Analysis
```

### 📊 전체 파이프라인 진행 상황

| Stage | Notebook | Status | 시간 | 비용 |
|-------|----------|--------|------|------|
| 데이터 생성 | 01-04 | ✅ 완료 | - | Free (T4) |
| LoRA SFT | 05 | ✅ 완료 | 8.2분 | 0.73 units |
| Prompt Tuning | 05b | ✅ 완료 | 18.8분 | 1.68 units |
| DPO | 06 | ⏳ 다음 | 예상 1-2시간 | 예상 5-10 units |
| Benchmark | 07 | ⏳ 대기 | - | - |
| Agent Eval | 08 | ⏳ 대기 | - | - |
| Analysis | 09 | ⏳ 대기 | - | - |

---

## 💡 Tips & Best Practices

### **비용 절감 전략**
1. ✅ 데이터 생성은 무료 T4 사용
2. ✅ Fine-tuning만 A100 사용 (실제로 10분 내외면 충분)
3. ✅ 불필요한 셀 재실행 방지
4. ✅ 학습 완료 후 즉시 런타임 종료

### **성능 최적화**
1. ✅ A100에서는 BF16 필수 사용
2. ✅ Batch size를 16까지 올릴 수 있음 (메모리 여유 시)
3. ✅ Gradient accumulation을 1로 줄이면 더 빠름
4. ⚠️ 단, effective batch size 유지 필요

### **안정성 확보**
1. ✅ BF16은 FP16보다 수치적으로 안정적
2. ✅ Checkpoint 시스템으로 중단 복구 가능
3. ✅ Gradient explosion 발생 시 learning rate 줄이기
4. ✅ 정기적으로 메모리 사용량 모니터링

### **실전 경험 공유**

**예상했던 것:**
- 학습 시간: 2-4시간
- 비용: 10-21 units
- 체크포인트: 2-3회

**실제 결과:**
- 학습 시간: **8.2분** (예상 대비 15-30배 빠름!)
- 비용: **0.73 units** (예상 대비 15배 저렴!)
- 체크포인트: 2회 (step 100, 114)

**교훈:**
1. A100 성능이 예상보다 훨씬 우수
2. 데이터셋 크기(900 samples)가 작아서 빠르게 완료
3. BF16 + TF32 최적화 효과가 큼
4. 100 units 구매는 충분히 여유 있음

---

## 🎓 LoRA vs Full Fine-Tuning 비교

| 항목 | Full Fine-Tuning | LoRA (r=8) | 비율 |
|------|------------------|------------|------|
| Trainable Params | 1.8B | 12.16M | **0.67%** |
| Memory | ~24GB | ~6GB | **25%** |
| Training Time | 10-20시간 | 8.2분 | **1%** |
| Model Size | ~7GB | ~50MB | **0.7%** |
| Quality | 100% | ~95-98% | 우수 |

**결론**: LoRA는 품질을 거의 유지하면서 비용/시간을 1% 수준으로 절감!

---

## 📚 참고 자료

### LoRA 논문
- **LoRA: Low-Rank Adaptation of Large Language Models** (2021)
- https://arxiv.org/abs/2106.09685

### 사용된 라이브러리
- **Transformers**: 4.41.0+
- **PEFT**: 0.7.0+ (LoRA 구현)
- **TRL**: 0.7.4+ (SFTTrainer)
- **BitsAndBytes**: 0.41.3+ (4-bit quantization)

---

**Status**: ✅ **COMPLETED on A100 GPU (2025-12-26)**
**Training Time**: 8.2 minutes
**Cost**: 0.73 compute units
**Eval Loss**: 0.541 (Excellent!)
**Next**: Notebook 05b (Prompt Tuning) or 06 (DPO)
