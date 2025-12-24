# Synthetic Instruction Tuner
## 프로젝트 요구사항 명세서

---

## 1. 과제 요구사항 (Course Project)

### 1.1 NLP Task 정의

**Task**: Instruction-Following Text Generation for Multi-Domain Dialogue Systems

본 프로젝트는 다양한 도메인(코딩, 추론, 창작 등)에서 고품질 응답을 생성할 수 있는 instruction-following LLM을 개발합니다.

### 1.2 필수 조건

**비교 분석 요구사항**:
- ✅ **다중 적응 방법 비교**: 단일 LLM에 여러 적응 전략 적용 OR 2개 이상의 LLM 비교
- ✅ **효율성 분석 포함**: 메모리 사용량, 학습 시간, 학습 가능 파라미터 수
- ✅ **성능 평가**: 벤치마크를 통한 정량적 비교

**제출물**:
1. ✅ **Code Repository**: 재현 가능한 코드, 문서화
2. ✅ **Written Report**: Introduction, Methods, Results, Comparative Analysis
3. ✅ **15-min Presentation**: 주요 발견 사항 발표

### 1.3 본 프로젝트의 접근

**비교 방법 선택**:
- 단일 베이스 모델 (Llama-3.2-3B)에 3가지 적응 방법 적용
  1. **LoRA (r=8)**: Low-Rank Adaptation
  2. **Prompt Tuning**: Soft prompt embeddings
  3. **DPO**: Direct Preference Optimization

**비교 메트릭**:
| 메트릭 | 측정 방법 |
|--------|----------|
| Trainable Parameters | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Peak Memory Usage | `torch.cuda.memory_allocated()` 최대값 |
| Training Time | 학습 시작~종료 시간 측정 |
| Inference Speed | 토큰/초 생성 속도 |
| Benchmark Scores | lm-evaluation-harness |

---

## 2. 기능 요구사항

### 2.1 데이터 생성 (Magpie Method)

**목표**: 고품질 합성 instruction 데이터 생성

**데이터 규모**:
- Raw 생성: 1,500개
- 품질 필터링 후: 1,000개
- Preference pairs: 600개

**기능**:
1. Llama-3.1-8B-Instruct를 사용한 템플릿 기반 생성
2. Instruction-response 페어 생성
3. JSON 형식으로 저장

**검증 기준**:
- Instruction 평균 길이: 20-150 단어
- Response 평균 길이: 50-500 단어
- 생성 성공률: 95% 이상

### 2.2 품질 필터링

**목표**: 1,500개 → 1,000개 고품질 데이터

**필터 규칙**:
1. **길이 필터**: 20-500 단어
2. **반복 필터**: 3회 이상 연속 반복 제거
3. **다양성 필터**: Jaccard 유사도 < 0.8
4. **거부 응답 필터**: "I'm an AI", "I cannot" 등 제거
5. **언어 필터**: 영어만 유지

**검증 기준**:
- 필터링 후 데이터: 1,000개 이상
- 품질 점수 평균: 0.7 이상

### 2.3 선호 데이터 생성

**목표**: Preference pairs 600개 생성

**프로세스**:
1. 필터링된 1,000개 instruction에서 샘플링
2. Instruction당 3개 응답 생성 (다양한 모델 사용)
3. Reward Model로 점수화
4. 최고점/최저점 쌍을 chosen/rejected로 선택

**검증 기준**:
- Chosen score > Rejected score
- Score 차이 평균: 0.15 이상

### 2.4 모델 미세조정

#### 2.4.1 LoRA SFT

**목표**: Parameter-efficient fine-tuning

**하이퍼파라미터**:
```python
{
    "r": 8,
    "lora_alpha": 16,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "trainable_params": ~0.1-0.5%
}
```

**검증 기준**:
- Training loss 수렴
- Eval loss < 1.8

#### 2.4.2 Prompt Tuning

**목표**: 최소 파라미터로 학습

**하이퍼파라미터**:
```python
{
    "num_virtual_tokens": 20,
    "init_method": "random",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 3e-4,
    "trainable_params": ~0.003%
}
```

**검증 기준**:
- Training loss 수렴
- 메모리 사용량 < LoRA

#### 2.4.3 DPO Training

**목표**: Preference alignment

**하이퍼파라미터**:
```python
{
    "beta": 0.1,
    "epochs": 1,
    "batch_size": 2,
    "learning_rate": 5e-5
}
```

**검증 기준**:
- Preference accuracy > 70%
- TruthfulQA 점수 향상

### 2.5 평가 시스템

#### 2.5.1 벤치마크 평가

**필수 벤치마크**:
1. MMLU (지식)
2. HellaSwag (상식 추론)
3. ARC-Easy (과학 추론)
4. TruthfulQA (진실성)

**평가 기준**:
- 모든 방법에 대해 동일 벤치마크 적용
- Zero-shot baseline과 비교

#### 2.5.2 비교 분석

**필수 항목**:
1. 효율성 비교 차트
2. 성능 vs 효율성 trade-off 분석
3. 사용 사례별 권장 방법

---

## 3. 비기능 요구사항

### 3.1 성능 요구사항

| 요구사항 | T4 목표 | A100 목표 |
|---------|---------|-----------|
| 데이터 생성 속도 | 1,500개 in 16-17h (3일 분할) | 1,500개 in 6-8h (1일) |
| SFT 학습 시간 | 3B model < 8h | 3B model < 4h |
| DPO 학습 시간 | 3B model < 6h | 3B model < 2h |
| 평가 실행 시간 | 모델당 < 4h | 모델당 < 3h |

### 3.2 리소스 제약

**하드웨어 제약 (T4)**:
- GPU: T4 (15GB VRAM)
- Runtime: 최대 12시간 연속
- 예산: $0 (완전 무료)

**하드웨어 제약 (A100 - 선택사항)**:
- GPU: A100 (40GB VRAM)
- Runtime: 무제한 (Colab Pro+) / 24시간 (Colab Pro)
- 예산: $10/월

**최적화 전략**:
- 4-bit quantization (NF4)
- Gradient accumulation
- Checkpoint 기반 재시작

### 3.3 재현성 요구사항

**코드**:
- Random seed 고정
- 모든 하이퍼파라미터 기록
- 실험 설정 문서화

**데이터**:
- 생성 데이터 저장
- 필터링 전후 버전 유지
- 데이터 통계 기록

### 3.4 문서화 요구사항

**코드 문서**:
- 모든 함수에 docstring
- README.md 완비
- 노트북에 설명 추가

**보고서**:
- Introduction & Dataset
- Model & Adaptation Methods
- Training Process
- Evaluation Results
- **Comparative Analysis** (필수)
- Conclusion

**발표 자료**:
- 15분 분량
- 주요 차트/표 포함
- 비교 분석 강조

---

## 4. 데이터 요구사항

### 4.1 데이터 형식

**SFT 데이터**:
```json
{
  "instruction": "string",
  "output": "string"
}
```

**DPO 데이터**:
```json
{
  "prompt": "string",
  "chosen": "string",
  "rejected": "string"
}
```

### 4.2 데이터 규모

| 단계 | 목표 규모 |
|------|-----------|
| Raw 생성 | 1,500 |
| 필터링 후 | 1,000 |
| SFT train/val | 640 / 160 |
| Preference pairs | 600 |
| DPO train/val | 400 / 100 |

### 4.3 데이터 품질 기준

**Instruction 품질**:
- 명확성: 요청이 명확해야 함
- 다양성: 주제/형식 다양
- 적절성: 유해 콘텐츠 없음

**Response 품질**:
- 정확성: 사실 기반
- 완전성: 충분한 설명
- 유용성: 실질적 도움

---

## 5. 시스템 요구사항

### 5.1 개발 환경

**필수 소프트웨어**:
- Python 3.10+
- CUDA 11.8+
- Google Colab account
- Hugging Face account

**필수 라이브러리**:
```
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.4
datasets>=2.16.0
accelerate>=0.25.0
bitsandbytes>=0.41.3
torch>=2.0.0
```

### 5.2 스토리지

**Google Drive 구조**:
```
synthetic-instruction-tuner/
├── data/
│   ├── raw/              # 1.5K samples (~10MB)
│   ├── filtered/         # 1K samples (~7MB)
│   └── preference/       # 600 pairs (~5MB)
├── models/
│   ├── sft/              # ~2GB per method
│   ├── prompt_tuning/    # ~100MB
│   └── dpo/              # ~2GB
└── evaluation/
    ├── results/          # ~50MB
    ├── figures/          # ~20MB
    └── metrics/          # ~10MB
```

**총 용량**: 약 15-20GB

---

## 6. 테스트 요구사항

### 6.1 단위 테스트

**데이터 생성**:
- Template 포맷 검증
- 생성 길이 검증
- JSON 직렬화 검증

**필터링**:
- 각 필터 규칙 검증
- 통계 계산 검증

**학습**:
- 데이터 로딩 검증
- 메모리 사용량 검증
- 체크포인트 저장/로딩 검증

### 6.2 통합 테스트

**End-to-end 파이프라인**:
1. 소규모 데이터 (100개)로 전체 파이프라인 실행
2. 각 단계 출력 검증
3. 최종 모델 추론 테스트

---

## 7. 제약사항 및 가정

### 7.1 제약사항

**리소스**:
- 무료 Colab만 사용
- T4 GPU 12시간 제한
- 메모리 15GB 상한

**시간**:
- 4주 완료
- 주당 20-30시간 투입

**데이터**:
- 합성 데이터만 사용 (인간 라벨 없음)
- 영어만 지원

### 7.2 가정

**모델 접근**:
- Llama 라이센스 동의 가능
- Hugging Face Hub 접근 가능

**성능**:
- T4 GPU로 3B 모델 학습 가능
- Colab 연결 끊김 시 재시작 가능

---

## 8. 우선순위

### 8.1 필수 (Must Have)

- ✅ LoRA SFT 구현 및 학습
- ✅ Prompt Tuning 구현 및 학습
- ✅ 효율성 메트릭 수집
- ✅ 비교 분석 노트북
- ✅ Written Report
- ✅ Presentation

### 8.2 중요 (Should Have)

- ✅ DPO 학습
- ✅ 벤치마크 평가
- 📝 AgentBench 평가 (선택)

### 8.3 선택 (Nice to Have)

- 📝 다른 PEFT 방법 (Adapters, Prefix-Tuning)
- 📝 더 큰 모델 실험 (7B)
- 📝 Human evaluation

---

## 9. 검증 체크리스트

### 9.1 데이터 검증
- [ ] Raw 데이터 1,500개 생성 완료
- [ ] 품질 필터링 후 1,000개 이상
- [ ] Preference pairs 600개 이상
- [ ] 데이터 형식 검증 통과

### 9.2 학습 검증
- [ ] LoRA 학습 loss 수렴
- [ ] Prompt Tuning 학습 loss 수렴
- [ ] DPO 학습 완료
- [ ] 모든 체크포인트 저장

### 9.3 평가 검증
- [ ] 벤치마크 평가 완료 (3개 방법)
- [ ] 효율성 메트릭 수집 완료
- [ ] 비교 분석 차트 생성
- [ ] 통계적 유의성 확인

### 9.4 문서 검증
- [ ] README 완성
- [ ] Written Report 완성 (모든 섹션)
- [ ] Presentation 완성 (15분)
- [ ] 코드 주석 완성

---

## 10. 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 2.0 | 2025-12-12 | 과제 요구사항 반영, 비교 분석 추가 |
| 1.0 | TBD | 초기 작성 |

---

*본 명세서는 프로젝트 진행에 따라 업데이트됩니다.*
