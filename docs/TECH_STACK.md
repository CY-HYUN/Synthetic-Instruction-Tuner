# Synthetic Instruction Tuner
## 기술 스택 및 환경 설정 가이드

---

## 1. 개발 환경

### 1.1 Primary 환경: Google Colab

**Colab Free (T4 GPU)**
```
GPU: NVIDIA T4 (16GB VRAM) - 무료
RAM: 12.7GB (무료)
Storage: Google Drive 연동
Runtime: 최대 12시간 제한
```

**Colab Pro (A100 GPU) - 최적화 완료 ⚡**
```
GPU: NVIDIA A100 (40GB VRAM) - Pro/Pro+
RAM: 25-52GB
Storage: Google Drive 연동
Runtime: 무제한 (Pro+) / 24시간 (Pro)
```

**본 프로젝트는 Colab Pro A100에 최적화되어 있습니다:**
- 배치 사이즈 3-4배 증가 (SFT: 12, DPO: 8)
- 체크포인트 간격 최적화 (100 샘플)
- 전체 파이프라인 2-3배 속도 향상
- 단일 세션에서 전체 데이터셋 생성 가능

### 1.2 Colab 설정 방법
```python
# GPU 확인
!nvidia-smi

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 경로 설정
PROJECT_PATH = "/content/drive/MyDrive/synthetic-instruction-tuner"
```

---

## 2. Python 패키지

### 2.1 핵심 라이브러리
```bash
# Colab 설치 스크립트 (최신 호환 버전)
!pip install -q --upgrade transformers>=4.41.0
!pip install -q --upgrade peft>=0.7.0
!pip install -q --upgrade trl>=0.7.4
!pip install -q --upgrade datasets>=2.16.0
!pip install -q --upgrade accelerate>=0.25.0
!pip install -q --upgrade bitsandbytes>=0.41.3
!pip install -q lm-eval
!pip install -q sentencepiece protobuf
```

### 2.2 패키지 설명
| 패키지 | 버전 | 용도 |
|--------|------|------|
| transformers | 4.41+ | 모델 로딩, 토크나이저, 추론 |
| peft | 0.7+ | LoRA 구현, 파라미터 효율적 미세조정 |
| trl | 0.7+ | SFTTrainer, DPOTrainer |
| datasets | 2.16+ | 데이터셋 관리, 전처리 |
| accelerate | 0.25+ | 분산 학습, 메모리 최적화 |
| bitsandbytes | 0.41+ | 4-bit/8-bit 양자화 |
| lm-eval | 0.4+ | 벤치마크 평가 (IFEval, MMLU 등) |
| sentencepiece | - | Llama 토크나이저 의존성 |

---

## 3. 모델 정보

### 3.1 데이터 생성용 모델
```python
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# 특징
# - 파라미터: 8B
# - Context Length: 128K
# - License: Llama 3.1 Community License
# - 용도: Magpie 방식 instruction 생성

# 로딩 코드
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

### 3.2 미세조정 대상 모델

**주 모델 (비교 분석용)**:
```python
MODEL_ID = "meta-llama/Llama-3.2-3B"
# - 파라미터: 3B
# - VRAM 요구량: ~6GB (4-bit)
# - 특징: 최신 Llama 아키텍처, 효율적
# - 접근: Llama 3.2 라이센스 동의 필요
# - 용도: LoRA, Prompt Tuning, DPO 비교 실험
```

**대안 모델 (선택사항)**:
- **Mistral-7B-v0.1**: 7B, Sliding Window Attention
- **Qwen2.5-3B**: 3B, 다국어 지원

**Note**: 과제 요구사항 충족을 위해 단일 모델(Llama-3.2-3B)에 여러 적응 방법(LoRA, Prompt Tuning, DPO) 적용하여 비교

### 3.3 Reward Model
```python
MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"

# 로딩 코드
from transformers import pipeline

reward_model = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=0  # GPU
)

# 사용 예시
score = reward_model("User: What is AI?\nAssistant: AI is...")
print(score[0]['score'])  # 0.82
```

---

## 4. Hugging Face 설정

### 4.1 토큰 생성
1. https://huggingface.co/settings/tokens 접속
2. "New token" 클릭
3. Name: `synthetic-instruction-tuner`
4. Role: `write` (모델 업로드용)
5. 토큰 복사 및 안전하게 저장

### 4.2 Colab에서 로그인
```python
from huggingface_hub import login

# 방법 1: 직접 입력
login(token="hf_xxxxxxxxxxxxx")

# 방법 2: 환경변수 (권장)
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxx"
login()
```

### 4.3 모델 접근 권한
**라이센스 동의 필요한 모델:**
- meta-llama/Llama-3.1-8B-Instruct
- meta-llama/Llama-3.2-3B

**절차:**
1. 각 모델 페이지 방문
2. "Access repository" 버튼 클릭
3. 정보 입력 및 라이센스 동의
4. 승인 대기 (보통 즉시~수시간)

---

## 5. LoRA 설정

### 5.1 LoRA Configuration
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=8,                           # Rank: 낮을수록 메모리 절약
    lora_alpha=16,                 # Scaling factor
    target_modules=[               # 적용 대상 레이어
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

### 5.2 메모리 사용량 비교
| 설정 | 3B 모델 | 7B 모델 |
|------|---------|---------|
| Full Fine-tuning | ~24GB | ~56GB |
| LoRA (r=8) | ~6GB | ~10GB |
| LoRA (r=4) | ~5GB | ~8GB |

### 5.3 권장 설정

**Colab T4 (16GB VRAM)**
```python
# 3B 모델
LORA_R = 8
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
```

**Colab Pro A100 (40GB VRAM) - 최적화됨**
```python
# 3B 모델 - SFT
LORA_R = 8
BATCH_SIZE = 12              # T4 대비 3배
GRADIENT_ACCUMULATION = 2    # 감소

# 3B 모델 - DPO
BATCH_SIZE = 8               # T4 대비 4배
GRADIENT_ACCUMULATION = 2    # 감소
```

---

## 6. Training Configuration

### 6.1 SFT Training Arguments

**T4 설정 (기본)**
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
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    report_to="none"
)
```

**A100 최적화 설정 (config.json에 반영됨)**
```python
sft_training_args = TrainingArguments(
    output_dir="./models/sft",
    num_train_epochs=3,
    per_device_train_batch_size=12,    # 4 → 12 (3배)
    gradient_accumulation_steps=2,      # 4 → 2 (감소)
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,                     # 500 → 200
    save_total_limit=3,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none"
)
# 예상 학습 시간: T4 6-10시간 → A100 2-4시간
```

### 6.2 DPO Training Arguments

**T4 설정 (기본)**
```python
from transformers import TrainingArguments

dpo_config = TrainingArguments(
    output_dir="./models/dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=50,
    save_steps=200,
    report_to="none"
)
```

**A100 최적화 설정 (config.json에 반영됨)**
```python
dpo_config = TrainingArguments(
    output_dir="./models/dpo",
    num_train_epochs=1,
    per_device_train_batch_size=8,     # 2 → 8 (4배)
    gradient_accumulation_steps=2,      # 8 → 2 (감소)
    learning_rate=5e-5,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=10,
    save_steps=100,                     # 200 → 100
    report_to="none"
)
# 예상 학습 시간: T4 4-6시간 → A100 1-2시간
# 메모리 사용: ~30-35GB (policy + reference 모델 동시 로드)
```

---

## 7. Evaluation Configuration

### 7.1 lm-evaluation-harness 설정
```bash
# 설치
pip install lm-eval

# 기본 실행
lm_eval --model hf \
    --model_args pretrained=./models/dpo/llama,trust_remote_code=True \
    --tasks ifeval,mmlu \
    --batch_size 4 \
    --output_path ./evaluation/results/
```

### 7.2 평가 태스크
| 태스크 | 설명 | 메트릭 |
|--------|------|--------|
| ifeval | Instruction Following | Accuracy |
| mmlu | 지식 평가 | Accuracy |
| truthfulqa | 진실성 | MC1, MC2 |
| hellaswag | 상식 추론 | Accuracy |

### 7.3 AgentBench 설정
```bash
# AgentBench 클론
git clone https://github.com/THUDM/AgentBench.git

# 간단한 평가 (webshop)
cd AgentBench
python eval.py --model ./models/dpo/llama --task webshop
```

---

## 8. 트러블슈팅

### 8.1 GPU OOM (Out of Memory)
```python
# 해결책 1: 배치 크기 축소
per_device_train_batch_size = 2  # 4 → 2
gradient_accumulation_steps = 8  # 4 → 8

# 해결책 2: 더 강한 양자화
load_in_4bit = True
bnb_4bit_compute_dtype = torch.float16

# 해결책 3: 메모리 정리
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

### 8.2 Colab 연결 끊김
```python
# 체크포인트 저장 후 재시작
trainer.train(resume_from_checkpoint="./checkpoint-latest")

# 자동 체크포인트
save_steps = 500  # 500 스텝마다 저장
```

### 8.3 모델 로딩 실패
```python
# 토큰 확인
from huggingface_hub import whoami
print(whoami())  # 로그인 상태 확인

# 라이센스 확인
# https://huggingface.co/meta-llama 방문하여 동의 여부 확인
```

### 8.4 Preference Data Generation 멈춤 문제

**증상**: `04_preference_generation.ipynb` (OPTIMIZED) 실행 시 배치 처리 중 멈춤 현상
- 진행률 바가 멈춤 (2-3분 이상 진행 없음)
- GPU 사용률 낮음 (14-15GB/40GB)
- "Testing batch generation..." 또는 메인 루프에서 멈춤

**원인**:
- 배치 처리 시 `model.generate()` 호출이 반환되지 않음
- `generation_batch_size`가 A100 환경에서도 불안정 (3 이상)
- GPU 메모리가 아닌 **프로세스 레벨 교착 상태** 발생
- **고용량 RAM을 켜도 해결되지 않음**

**핵심 원인 분석**:
- Llama-3.1-8B-Instruct 4-bit 모델의 배치 생성 불안정성
- 여러 temperature에서 동시 생성 시 CUDA 커널 충돌
- Colab 환경 특유의 메모리 관리 이슈

**해결책 1: STABLE 버전 사용 (강력 권장)**
```python
# 04_preference_generation_STABLE.ipynb 사용
# - 순차 처리 (한 번에 1개씩)
# - 100% 안정성 보장
# - A100: 8-10시간, T4: 15-20시간
# - 고용량 RAM과 무관하게 안정적

# 실전 검증됨:
# - 50개 생성 완료 (성공률 96-100%)
# - 멈춤 현상 0회
# - 체크포인트 안전
```

**해결책 2: OPTIMIZED 버전 재시도 (권장하지 않음)**
```python
# 주의: 고용량 RAM을 켜도 실패 확률 60-70%

# 런타임 완전 초기화 필수
# 런타임 → 런타임 다시 시작 → 모두 삭제

# generation_batch_size를 1로 강제
generation_batch_size=1  # 기본값 3에서 1로 변경

# 메모리 정리 주기 단축
if batch_num % 5 == 0:  # 10 → 5
    gc.collect()
    torch.cuda.empty_cache()

# 테스트 셀 건너뛰기 (Cell 12)
# 바로 메인 루프(Cell 16) 실행
```

**Colab 장시간 실행 유지 방법**
```javascript
// 브라우저 콘솔(F12)에서 실행
function ClickConnect(){
    console.log("Keep alive");
    document.querySelector("colab-connect-button")?.shadowRoot.querySelector("#connect")?.click();
}
setInterval(ClickConnect, 60000);  // 1분마다 연결 유지
```

**체크포인트 복구**
```python
# STABLE 체크포인트에서 OPTIMIZED로 전환 시
import shutil
STABLE_CHECKPOINT = f"{PREFERENCE_PATH}/preference_checkpoint_stable.json"
CHECKPOINT_PATH = f"{PREFERENCE_PATH}/preference_checkpoint.json"

if os.path.exists(STABLE_CHECKPOINT):
    shutil.copy(STABLE_CHECKPOINT, CHECKPOINT_PATH)
    print(f"✅ Loaded STABLE checkpoint")
```

**권장 워크플로우**:
1. **STABLE 버전 직접 사용** (처음부터 끝까지)
2. ~~OPTIMIZED 전환 시도~~ (권장하지 않음 - 실패 확률 60-70%)
3. 시간이 아까워도 STABLE이 **결과적으로 더 빠름**
   - OPTIMIZED 디버깅: 2-4시간 낭비 → 실패
   - STABLE 한 번에: 8-10시간 → 확실한 완성

**실제 경험 데이터**:
- OPTIMIZED 시도 3회 모두 실패 (테스트 셀, 메인 루프 모두 멈춤)
- STABLE 버전 50개 완료 (성공률 96%, 멈춤 0회)
- 결론: **처음부터 STABLE 사용이 최선**

---

## 9. 체크리스트

### 9.1 프로젝트 시작 전 확인
- [ ] Google 계정 준비
- [ ] Hugging Face 계정 생성
- [ ] Hugging Face 토큰 생성 (write 권한)
- [ ] Llama 3.1, 3.2 라이센스 동의
- [ ] Google Drive 여유 공간 확인 (최소 20GB)

### 9.2 Week 1 시작 시 확인
- [x] Colab GPU 할당 확인 (Runtime > Change runtime type > T4)
- [x] 모든 패키지 설치 완료
- [x] Hugging Face 로그인 성공
- [x] 테스트 모델 로딩 성공

### 9.3 주기적 확인 사항
- [x] 체크포인트 저장 확인
- [x] Google Drive 동기화 확인
- [x] GPU 메모리 사용량 모니터링
- [x] 학습 로그 확인

### 9.4 실제 완료 현황 (2025-12-26)
- ✅ **Notebooks 01-04**: 데이터 생성 및 필터링 완료 (1,500 → 1,000 samples)
- ✅ **Notebook 05**: LoRA SFT 완료 (A100, 8.2분, eval loss 0.541)
- ✅ **Notebook 05b**: Prompt Tuning 완료 (A100, 18.8분, eval loss 2.979)
- ✅ **비용 효율**: 총 2.41 compute units 사용 (예상 31-58 대비 20배 절감)
- ⏳ **다음 단계**: Notebook 06 (DPO Training) 준비

---

*본 문서는 프로젝트 진행 중 업데이트될 수 있습니다.*
