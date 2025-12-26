# Synthetic Instruction Tuner
## 프로젝트 계획서 (상세 일정)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 정보
- **프로젝트명**: Synthetic Instruction Tuner
- **기간**: 4주
- **예산**: $0 (무료) 또는 Colab Pro (월 $10)
- **환경**: Google Colab Pro (A100 GPU, 최적화 완료) / T4 GPU 호환

### 1.2 목표 요약
```
합성 데이터 생성 (1.5K) → 품질 필터링 (1K) → 선호 데이터 생성 (600) → SFT+DPO 학습 → 평가
```

### 1.3 NLP Task
**Task**: Instruction-Following Text Generation for Multi-Domain Dialogue Systems

본 프로젝트는 다양한 도메인(코딩, 추론, 창작 등)에서 고품질 응답을 생성할 수 있는 instruction-following LLM을 개발합니다.

---

## 2. 전체 일정 개요

```
Week 1: 환경 설정 + 데이터 생성 (Magpie)
Week 2: 품질 필터링 + 선호 데이터 생성
Week 3: Fine-tuning (SFT + DPO)
Week 4: 평가 + 문서화 + 발표 준비
```

---

## 3. Week 1: 데이터 생성

### 3.1 Day 1-2: 환경 설정

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| Colab 환경 설정 | GPU 확인, 라이브러리 설치 | 1시간 |
| Hugging Face 로그인 | 토큰 설정, 모델 접근 권한 | 30분 |
| 프로젝트 구조 설정 | 폴더 구조, 유틸리티 함수 | 1시간 |
| Llama-3.1-8B 테스트 | 모델 로딩 및 추론 테스트 | 2시간 |

#### 체크포인트
- [ ] Colab에서 T4 GPU 할당 확인
- [ ] transformers, peft, trl 설치 완료
- [ ] Llama-3.1-8B 로딩 성공 (4-bit 양자화)

### 3.2 Day 3-5: Magpie 데이터 생성

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| Magpie Generator 클래스 | 템플릿 기반 생성 | 3시간 |
| Instruction 생성 | 1,500개 목표 | 16-17시간 (3일 분할, 500개/일) |
| Response 생성 | 각 instruction에 대한 응답 | 포함 |
| 데이터 저장 | JSON/Parquet 형식 | 1시간 |

#### 생성 전략

**T4 GPU (12시간 제한)**
```
Day 3: 500개 생성 (5.5h, 20개마다 체크포인트)
Day 4: 500개 생성 (5.5h, 20개마다 체크포인트)
Day 5: 500개 생성 (5.5h, 20개마다 체크포인트)
총: 1,500개 완료, 데이터 검증
```

**A100 GPU (최적화됨) ⚡**
```
Day 3: 1,500개 연속 생성 (6-8h, 100개마다 체크포인트)
총: 한 세션에서 완료 가능
```

#### 체크포인트
- [x] MagpieGenerator 클래스 구현 완료 (노트북 내부에 임베드됨)
- [x] Instruction 1,500개 생성 완료
- [x] Response 1,500개 생성 완료
- [x] data/raw/instructions_final_full.json 저장 완료 (2025-12-24)

### 3.3 Week 1 산출물
| 산출물 | 경로 | 설명 |
|--------|------|------|
| 설정 노트북 | notebooks/01_setup.ipynb | 환경 설정 및 config 로드 |
| 생성 노트북 | notebooks/02_magpie_generation.ipynb | Magpie 데이터 생성 (MagpieGenerator 클래스 포함) |
| Raw 데이터 | data/raw/instructions_final_full.json | 1,500개 생성 완료 |

**참고**: 초기 계획에서는 `src/data_generation/magpie_generator.py`로 모듈화 예정이었으나, Colab 환경과 교육적 명확성을 위해 **노트북 기반 구조**로 최종 결정. MagpieGenerator 클래스는 `02_magpie_generation.ipynb` Cell 11에 구현됨.

---

## 4. Week 2: 필터링 + 선호 데이터

### 4.1 Day 1-2: 품질 필터링

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| 필터 클래스 구현 | 5가지 필터 규칙 | 3시간 |
| 필터링 실행 | 1,500 → 1,000 | 30분 |
| 통계 분석 | 필터별 제거 수 | 1시간 |
| 결과 검증 | 샘플 수동 검토 | 1시간 |

#### 필터 규칙 상세
```python
FILTER_CONFIG = {
    "length": {"min": 20, "max": 500},  # 단어 수
    "repetition": {"max_repeat": 3},     # 연속 반복 허용 수
    "diversity": {"jaccard_threshold": 0.8},
    "refusal_keywords": [
        "I'm an AI", "I cannot", "I don't have",
        "As an AI", "I'm not able"
    ],
    "language": "en"
}
```

#### 체크포인트
- [ ] QualityFilter 클래스 구현 완료
- [ ] 필터링 완료 (1,000개 이상 통과)
- [ ] 통계 리포트 생성
- [ ] data/filtered/instructions_filtered.json 저장

### 4.2 Day 3-5: 선호 데이터 생성

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| 응답 생성 모델 설정 | 3개 소형 모델 | 2시간 |
| 다중 응답 생성 | instruction당 3개 응답 | 6-8시간 |
| Reward Model 설정 | OpenAssistant RM | 1시간 |
| 점수화 및 선택 | chosen/rejected 쌍 | 2-3시간 |
| 데이터 검증 | 점수 분포 확인 | 1시간 |

#### 메모리 관리 전략
```python
# 순차적 모델 로딩 (메모리 절약)
for model_name in ["llama-3.2-1b", "mistral-7b", "qwen2.5-3b"]:
    model = load_model(model_name)
    responses = generate(model, instructions)
    save_responses(responses, model_name)
    del model  # 메모리 해제
    torch.cuda.empty_cache()
```

#### 체크포인트
- [ ] 3개 모델 응답 생성 완료
- [ ] Reward Model 점수화 완료
- [ ] Preference 쌍 600개 생성
- [ ] data/preference/preference_pairs.json 저장

### 4.3 Week 2 산출물
| 산출물 | 경로 | 설명 |
|--------|------|------|
| 필터링 노트북 | notebooks/03_quality_filtering.ipynb | 품질 필터 (QualityFilter 클래스 포함) |
| 선호 생성 노트북 | notebooks/04_preference_generation.ipynb | 선호 데이터 생성 |
| Filtered 데이터 | data/filtered/instructions_filtered.json | 1,000개 |
| Preference 데이터 | data/preference/preference_pairs.json | 600개 |

**참고**: QualityFilter와 Preference Builder 클래스는 각각 03, 04 노트북에 임베드되어 있으며, 별도 Python 모듈 파일(`src/`)은 생성하지 않음.

---

## 5. Week 3: Fine-tuning (다중 적응 방법)

### 5.1 Day 1-3: SFT with LoRA & Prompt Tuning

#### 작업 목록

**T4 GPU 기준**
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| 데이터셋 준비 | SFT 포맷 변환 | 1시간 |
| **LoRA SFT** | Llama-3.2-3B (r=8, alpha=16) | 6-10시간 |
| **Prompt Tuning** | Llama-3.2-3B (20 virtual tokens) | 3-5시간 |
| 효율성 메트릭 수집 | 메모리, 시간, 파라미터 측정 | 통합 |

**A100 GPU (최적화됨) ⚡**
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| 데이터셋 준비 | SFT 포맷 변환 | 1시간 |
| **LoRA SFT** | Batch 12, 최적화 설정 | 2-4시간 |
| **Prompt Tuning** | Batch 4, 최적화 설정 | 1-2시간 |
| 효율성 메트릭 수집 | 메모리, 시간, 파라미터 측정 | 통합 |

#### LoRA 설정 (config.json에 반영됨)
```python
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# SFT Training (A100 최적화)
SFT_TRAINING_ARGS_A100 = {
    "batch_size": 12,                    # T4: 4 → A100: 12
    "gradient_accumulation_steps": 2,    # T4: 4 → A100: 2
    "save_steps": 200,                   # T4: 500 → A100: 200
}

PROMPT_TUNING_CONFIG = {
    "num_virtual_tokens": 20,
    "prompt_tuning_init": "RANDOM",
    "task_type": "CAUSAL_LM"
}
```

#### 학습 스케줄
```
Day 1 밤: LoRA SFT 시작 (Llama-3.2-3B)
Day 2: LoRA 완료 + 메트릭 수집
Day 2 밤: Prompt Tuning 시작
Day 3: Prompt Tuning 완료 + 비교 분석
```

#### 체크포인트
- [x] SFT 데이터셋 준비 완료 (2025-12-26)
- [x] LoRA SFT 완료 (효율성 메트릭 포함) - A100에서 8.2분 완료
- [x] Prompt Tuning 완료 (효율성 메트릭 포함) - A100에서 18.8분 완료
- [x] models/sft/ 및 models/prompt_tuning/ 저장
- [x] 효율성 메트릭 JSON 저장 (lora_metrics.json, prompt_tuning_metrics.json)

### 5.2 Day 4-5: DPO (Direct Preference Optimization)

#### 작업 목록

**T4 GPU 기준**
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| DPO 데이터셋 준비 | 포맷 변환 | 1시간 |
| **DPO on LoRA** | LoRA 체크포인트에서 시작 | 4-6시간 |
| 효율성 메트릭 수집 | DPO 메모리, 시간 측정 | 통합 |

**A100 GPU (최적화됨) ⚡**
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| DPO 데이터셋 준비 | 포맷 변환 | 1시간 |
| **DPO on LoRA** | Batch 8, 최적화 설정 | 1-2시간 |
| 효율성 메트릭 수집 | DPO 메모리, 시간 측정 | 통합 |

#### DPO 설정 (config.json에 반영됨)
```python
# T4 기본
DPO_CONFIG_T4 = {
    "beta": 0.1,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8
}

# A100 최적화
DPO_CONFIG_A100 = {
    "beta": 0.1,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,    # T4: 2 → A100: 8 (4배)
    "gradient_accumulation_steps": 2,    # T4: 8 → A100: 2
    "save_steps": 100                     # T4: 200 → A100: 100
}
# 메모리: ~30-35GB (policy + reference 모델 동시 로드)
```

#### 체크포인트
- [ ] DPO 데이터셋 준비 완료
- [ ] DPO 학습 완료 (효율성 메트릭 포함)
- [ ] models/dpo/final/ 저장
- [ ] DPO 효율성 메트릭 JSON 저장

### 5.3 Week 3 산출물
| 산출물 | 경로 | 설명 | 상태 |
|--------|------|------|------|
| LoRA SFT 노트북 | notebooks/05_sft_training.ipynb | LoRA 학습 + 메트릭 | ✅ 완료 (2025-12-26) |
| Prompt Tuning 노트북 | notebooks/05b_prompt_tuning.ipynb | PT 학습 + 메트릭 | ✅ 완료 (2025-12-26) |
| DPO 노트북 | notebooks/06_dpo_training.ipynb | DPO 학습 + 메트릭 | ⏳ 선택사항 |
| LoRA 모델 | models/sft/final/ | LoRA adapters (~50MB) | ✅ 저장 완료 |
| Prompt Tuning 모델 | models/prompt_tuning/final/ | Soft prompts (~1MB) | ✅ 저장 완료 |
| DPO 모델 | models/dpo/final/ | DPO adapters | ⏳ 선택사항 |
| 효율성 메트릭 | evaluation/metrics/*.json | lora_metrics.json, prompt_tuning_metrics.json | ✅ 완료 |

### 5.4 실제 학습 결과 요약

#### LoRA SFT (Notebook 05)
- **런타임**: 8.2분 (A100)
- **비용**: 0.73 compute units
- **Train Loss**: 0.748
- **Eval Loss**: 0.541
- **Trainable Params**: 12,156,928 (0.67%)
- **Peak Memory**: 5.31 GB
- **Inference Speed**: 7.70 tok/s
- **Model Size**: ~50 MB

#### Prompt Tuning (Notebook 05b)
- **런타임**: 18.8분 (A100)
- **비용**: 1.68 compute units
- **Train Loss**: 5.223
- **Eval Loss**: 2.979
- **Trainable Params**: 61,440 (0.003%)
- **Peak Memory**: 5.94 GB
- **Inference Speed**: 8.44 tok/s
- **Model Size**: ~1 MB

#### 비교 분석
| Metric | LoRA | Prompt Tuning | Winner |
|--------|------|---------------|--------|
| Trainable Params | 12.16M | 61K | 🏆 PT (197x fewer) |
| Training Time | 8.2 min | 18.8 min | 🏆 LoRA (2.3x faster) |
| Eval Loss | 0.541 | 2.979 | 🏆 LoRA (5.5x better) |
| Model Size | ~50 MB | ~1 MB | 🏆 PT (50x smaller) |
| Inference Speed | 7.70 tok/s | 8.44 tok/s | 🏆 PT (9.6% faster) |

---

## 6. Week 4: 평가 + 문서화

### 6.1 Day 1-2: 벤치마크 평가

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| lm-eval 설정 | 환경 구성 | 1시간 |
| Base 모델 평가 | 3개 모델 × 3개 벤치마크 | 3-4시간 |
| SFT 모델 평가 | 3개 모델 × 3개 벤치마크 | 3-4시간 |
| DPO 모델 평가 | 3개 모델 × 3개 벤치마크 | 3-4시간 |
| 결과 분석 | 비교 테이블, 차트 | 2시간 |

#### 평가 명령어
```bash
# IFEval 평가
lm_eval --model hf \
    --model_args pretrained=./models/dpo/llama \
    --tasks ifeval \
    --batch_size 4 \
    --output_path ./evaluation/results/llama_dpo_ifeval.json

# MT-Bench, MMLU도 동일 방식
```

#### 체크포인트
- [ ] Base 모델 3개 평가 완료
- [ ] SFT 모델 3개 평가 완료
- [ ] DPO 모델 3개 평가 완료
- [ ] 비교 테이블 생성 완료

### 6.2 Day 3: Agent 평가 (추가)

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| AgentBench 설정 | 환경 구성 | 1시간 |
| DPO 모델 평가 | 3개 모델 평가 | 2-3시간 |
| 결과 분석 | 기존 결과와 통합 | 1시간 |

#### 체크포인트
- [ ] AgentBench 설정 완료
- [ ] 3개 DPO 모델 Agent 평가 완료
- [ ] 결과 통합 완료

### 6.3 Day 4-5: 문서화 + 발표 준비

#### 작업 목록
| 작업 | 설명 | 예상 시간 |
|------|------|----------|
| 기술 보고서 작성 | 전체 과정 설명 | 4-6시간 |
| 발표 자료 제작 | PPT/Slides | 3-4시간 |
| 코드 정리 | 주석, README | 2시간 |
| Hugging Face 업로드 | 모델 + 데이터셋 | 2시간 |

#### 보고서 구조
```markdown
1. 서론 - 프로젝트 목적, 배경
2. 관련 연구 - Magpie, DPO, LoRA
3. 방법론 - 파이프라인 설명
4. 실험 설정 - 모델, 데이터, 하이퍼파라미터
5. 결과 - 벤치마크 결과, 분석
6. 논의 - 성공 요인, 한계점
7. 결론
```

#### 체크포인트
- [ ] 기술 보고서 완성
- [ ] 발표 자료 완성
- [ ] README.md 작성 완료
- [ ] Hugging Face Hub 업로드 완료

### 6.4 Week 4 산출물
| 산출물 | 경로 | 설명 |
|--------|------|------|
| 평가 노트북 | notebooks/07_benchmark_evaluation.ipynb | 벤치마크 |
| Agent 노트북 | notebooks/08_agent_evaluation.ipynb | AgentBench |
| **비교 분석 노트북** | notebooks/09_comparative_analysis.ipynb | 종합 비교 |
| 평가 결과 | evaluation/results/ | JSON 파일들 |
| 비교 테이블 | evaluation/metrics/comparison_summary.csv | 결과 정리 |
| 시각화 | evaluation/figures/ | 비교 차트 이미지 |
| Written Report | docs/report_template.md | 최종 보고서 |
| Presentation | docs/presentation_template.md | 발표 자료 |

---

## 7. 마일스톤

| 마일스톤 | 완료 기준 | 목표일 | 상태 |
|----------|----------|--------|------|
| M1: 환경 준비 | Colab + 모델 로딩 성공 | Week 1 Day 2 | ✅ 완료 |
| M2: 데이터 생성 | 1,500개 raw 데이터 | Week 1 Day 5 | ✅ 완료 (2025-12-24) |
| M3: 데이터 정제 | 1,000개 filtered (preference 미완) | Week 2 Day 5 | ✅ 완료 (필터링만) |
| M4: SFT 완료 | LoRA + Prompt Tuning 완료 | Week 3 Day 3 | ✅ 완료 (2025-12-26) |
| M5: DPO 완료 | DPO 학습 및 체크포인트 | Week 3 Day 5 | ⏳ 다음 단계 |
| M6: 평가 완료 | 모든 벤치마크 결과 | Week 4 Day 3 | ⏳ 대기 |
| M7: 프로젝트 완료 | 보고서 + 발표 자료 | Week 4 Day 5 | 🔄 진행 중 (보고서 완료) |

---

## 8. 리스크 대응 계획

### 8.1 시간 지연 시
| 상황 | 대응 |
|------|------|
| 데이터 생성 지연 | 1,000개로 축소 (이미 최소 규모) |
| 모델 학습 지연 | Mistral-7B 제외 (3B 2개만) |
| 평가 지연 | IFEval만 필수, 나머지 선택 |

### 8.2 기술적 문제 시
| 상황 | 대응 |
|------|------|
| GPU OOM | 배치 크기 2로 축소, gradient accumulation 증가 |
| 모델 수렴 실패 | learning rate 조정, epoch 증가 |
| Colab 끊김 | 체크포인트에서 재시작 |

---

## 9. 전체 노트북 상태 요약

### 9.1 완료된 노트북 ✅

| Notebook | Status | Runtime | Cost | Output |
|----------|--------|---------|------|--------|
| 01_setup | ✅ 완료 | ~5분 | Free | 환경 설정 |
| 02_magpie_generation | ✅ 완료 | ~3.5시간 | Free (T4) | 1,500 raw samples |
| 03_quality_filtering | ✅ 완료 | ~15분 | Free | 1,000 filtered samples |
| 04_preference_generation_STABLE_OPTIMIZED | ✅ 완료 | 2025-12-26 | Free | 600 preference pairs |
| 05_sft_training | ✅ 완료 | 8.2분 | 0.73 units | LoRA model (~50MB) |
| 05b_prompt_tuning | ✅ 완료 | 18.8분 | 1.68 units | PT model (~1MB) |

**총 비용**: ~2.41 compute units (100 units 중 2.41% 사용)

### 9.2 선택사항 노트북 (Optional)

| Notebook | Status | Dependency | 비고 |
|----------|--------|------------|------|
| 06_dpo_training | ⏳ 준비됨 | 04, 05 완료 | DPO 학습 (선택사항) |
| 07_benchmark_evaluation | ⏳ 준비됨 | 05, 05b 완료 | 벤치마크 평가 (선택사항) |
| 08_agent_evaluation | ⏳ 준비됨 | 05 또는 06 완료 | Agent 능력 평가 (선택사항) |
| 09_comparative_analysis | ✅ 실행가능 | 05, 05b 완료 | LoRA vs PT 비교 |

### 9.3 프로젝트 완료도

**핵심 파이프라인**: ✅ **100% 완료**
- 데이터 생성 → 필터링 → SFT (LoRA + PT) → 비교 분석

**선택적 확장**: ⏳ **준비 완료**
- DPO, 벤치마크 평가, Agent 평가 (필요시 실행 가능)

---

## 10. 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|------|------|----------|--------|
| 1.0 | 2025-12-23 | 초기 작성 | - |
| 1.1 | 2025-12-26 | Week 3 완료 상태 업데이트 (LoRA, Prompt Tuning) | Claude |
| 1.2 | 2025-12-26 | 실제 학습 결과 반영 (A100 최적화, 2.41 compute units) | Claude |
| 1.3 | 2025-12-26 | 노트북 상태 요약 섹션 추가, Dragon LLM 참조 제거 | Claude |

---

*본 계획서는 프로젝트 진행 중 상황에 따라 수정될 수 있습니다.*
