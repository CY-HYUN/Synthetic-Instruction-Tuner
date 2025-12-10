# Synthetic Instruction Tuner
## 프로젝트 요구사항 상세 설명서

---

## 1. 프로젝트 개요

### 1.1 프로젝트 명
**Synthetic Instruction Tuner**
> End-to-End Synthetic Data Generation and Fine-tuning Pipeline for Instruction-Tuned LLMs

### 1.2 프로젝트 목적
1. **학술적 목표**: LLM 과목 기말 프로젝트 요구사항 충족
2. **실무적 목표**: Dragon LLM 인턴십 (Synthetic Data Generation for Agentic LLMs) 준비
3. **기술적 목표**: 합성 데이터 생성부터 미세조정, 평가까지의 End-to-End 파이프라인 구축

### 1.3 프로젝트 기간
- **예상 기간**: 4주
- **시작일**: TBD
- **종료일**: TBD

### 1.4 예산
- **목표 예산**: $0 (완전 무료)
- **최대 허용 예산**: $10 (Colab Pro 1개월, 선택사항)

---

## 2. 프로젝트 배경

### 2.1 교수님 요구사항 (강의 내용 기반)
강의 녹취록에서 추출한 핵심 요구사항:

| 항목 | 요구사항 | 중요도 |
|------|----------|--------|
| NLP Task | 자유 선택 가능 | 필수 |
| LLM 활용 | 단계별 이해와 함께 구현 | 필수 |
| Fine-tuning | 미세조정 기법 적용 | 필수 |
| 평가 | 성능 평가 및 비교 | 필수 |
| 코드 품질 | 깔끔한 구조, 재현 가능성 | 필수 |
| 보고서 | 과정 설명 포함 | 필수 |
| 프레젠테이션 | 발표 자료 | 필수 |
| **프로세스 이해** | **결과보다 과정 이해 중시** | **핵심** |

### 2.2 Dragon LLM 인턴십 연관성
**직무**: Synthetic Data Generation for Agentic LLMs (6개월, 파리)

| 인턴십 요구사항 | 프로젝트 대응 |
|----------------|--------------|
| Synthetic instruction data pipeline | Magpie 방식 데이터 생성 |
| Reinforcement data pipeline | DPO를 통한 선호 학습 |
| Fine-tuning with RL | SFT + DPO 파이프라인 |
| lm-evaluation-harness | 표준 벤치마크 평가 |
| Hugging Face 경험 (필수) | transformers, TRL, PEFT 활용 |
| 결과 공개 | GitHub + Hugging Face Hub |

---

## 3. 기술 요구사항

### 3.1 사용 기술 스택

#### 3.1.1 프로그래밍 언어
- **Python 3.10+**

#### 3.1.2 핵심 라이브러리
| 라이브러리 | 용도 | 버전 |
|-----------|------|------|
| transformers | 모델 로딩/추론 | 4.36+ |
| peft | LoRA 미세조정 | 0.7+ |
| trl | SFT/DPO Trainer | 0.7+ |
| datasets | 데이터셋 관리 | 2.16+ |
| lm-eval | 벤치마크 평가 | 0.4+ |
| torch | 딥러닝 프레임워크 | 2.1+ |
| bitsandbytes | 양자화 | 0.41+ |
| accelerate | 분산 학습 | 0.25+ |

#### 3.1.3 개발 환경
- **Primary**: Google Colab (T4 GPU, 무료)
- **Backup**: Colab Pro (A100/V100, $10/월)
- **Local**: VSCode + Python

### 3.2 모델 요구사항

#### 3.2.1 데이터 생성용 모델
| 모델 | 용도 | 크기 |
|------|------|------|
| meta-llama/Llama-3.1-8B-Instruct | Magpie 데이터 생성 | 8B |

#### 3.2.2 미세조정 대상 모델 (3개)
| 모델 | 크기 | 선정 이유 |
|------|------|----------|
| meta-llama/Llama-3.2-3B | 3B | 최신 Llama, 효율적 |
| mistralai/Mistral-7B-v0.1 | 7B | 성능 우수, 커뮤니티 활발 |
| Qwen/Qwen2.5-3B | 3B | 다국어 지원, 다양성 |

#### 3.2.3 평가/보조 모델
| 모델 | 용도 | 비용 |
|------|------|------|
| OpenAssistant/reward-model-deberta-v3-large-v2 | 선호 데이터 점수화 | $0 |

### 3.3 데이터 요구사항

#### 3.3.1 생성 데이터 규모
| 데이터 유형 | 목표 수량 | 용도 |
|------------|----------|------|
| Raw Instructions | 15,000개 | 초기 생성 |
| Filtered Instructions | 10,000개 | 필터링 후 |
| Preference Pairs | 10,000개 | DPO 학습용 |

#### 3.3.2 데이터 품질 기준
- **길이**: 20~500 단어
- **다양성**: Jaccard 유사도 < 0.8
- **품질**: 반복 패턴 없음, 거부 응답 없음

---

## 4. 기능 요구사항

### 4.1 Phase 1: Synthetic Data Generation

#### 4.1.1 Magpie 방식 구현
```
입력: 템플릿 프롬프트만 제공
      "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"

처리: Llama-3.1-8B-Instruct가 자동으로 instruction 생성
      이어서 response도 생성

출력: {instruction, response} 쌍 15,000개
```

#### 4.1.2 구현 요구사항
- [ ] Llama-3.1-8B-Instruct 모델 로딩 (4-bit 양자화)
- [ ] 템플릿 기반 instruction 생성 함수
- [ ] Response 생성 함수
- [ ] 배치 처리 (메모리 효율)
- [ ] 체크포인트 저장 (1,000개마다)
- [ ] 진행률 표시

### 4.2 Phase 2: Quality Filtering

#### 4.2.1 규칙 기반 필터링
```
입력: 15,000개 raw instruction-response 쌍

필터:
1. 길이 검증 (20-500 단어)
2. 반복 패턴 감지
3. 다양성 검사 (Jaccard < 0.8)
4. 거부 응답 필터 ("I'm an AI", "I cannot" 등)
5. 언어 일관성 (영어만)

출력: 10,000개 고품질 데이터
```

#### 4.2.2 구현 요구사항
- [ ] 길이 필터 함수
- [ ] 반복 패턴 감지 함수 (정규표현식)
- [ ] Jaccard 유사도 계산 함수
- [ ] 키워드 블랙리스트 필터
- [ ] 통계 리포트 생성 (필터별 제거 수)

### 4.3 Phase 3: Preference Data Generation

#### 4.3.1 다중 모델 응답 생성
```
입력: 10,000개 instruction

처리:
1. 각 instruction에 대해 3개 모델이 응답 생성
2. OpenAssistant Reward Model로 점수 매김
3. 최고 점수 vs 최저 점수 선택

출력: {instruction, chosen, rejected} 쌍 10,000개
```

#### 4.3.2 구현 요구사항
- [ ] 3개 모델 로딩 함수 (순차적, 메모리 관리)
- [ ] 응답 생성 함수
- [ ] Reward Model 점수화 함수
- [ ] Preference pair 생성 함수
- [ ] 점수 분포 분석

### 4.4 Phase 4: Fine-tuning

#### 4.4.1 SFT (Supervised Fine-Tuning)
```
입력: 10,000개 instruction-response 쌍
모델: 3개 base 모델 각각

설정:
- LoRA: r=8, alpha=16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Epochs: 3
- Batch size: 4 (gradient accumulation: 4)
- Learning rate: 2e-4

출력: 3개 SFT 모델 체크포인트
```

#### 4.4.2 DPO (Direct Preference Optimization)
```
입력: 10,000개 preference 쌍
모델: 3개 SFT 체크포인트

설정:
- Beta: 0.1
- Learning rate: 5e-5
- Epochs: 1

출력: 3개 SFT+DPO 모델 체크포인트
```

#### 4.4.3 구현 요구사항
- [ ] LoRA 설정 함수
- [ ] SFTTrainer 설정 및 학습
- [ ] DPOTrainer 설정 및 학습
- [ ] 체크포인트 저장/로딩
- [ ] 학습 로그 기록 (wandb 선택사항)
- [ ] OOM 방지 설정

### 4.5 Phase 5: Evaluation

#### 4.5.1 표준 벤치마크 평가
```
평가 대상:
- 3개 Base 모델 (비교 기준)
- 3개 SFT 모델
- 3개 SFT+DPO 모델

벤치마크:
- IFEval: Instruction Following 능력
- MT-Bench: 대화 품질
- MMLU: 지식 이해도

도구: lm-evaluation-harness
```

#### 4.5.2 Agent 평가 (추가)
```
평가 대상: 최종 SFT+DPO 모델 3개

벤치마크:
- AgentBench (webshop 서브셋)

목적: Agentic LLM 능력 검증 (인턴십 어필)
```

#### 4.5.3 구현 요구사항
- [ ] lm-eval 설정 스크립트
- [ ] 결과 파싱 함수
- [ ] 비교 테이블 생성
- [ ] 시각화 (차트)
- [ ] AgentBench 설정

---

## 5. 비기능 요구사항

### 5.1 성능 요구사항
- 전체 파이프라인 4주 내 완료
- Colab 무료 버전으로 실행 가능
- 12시간 제한 내 체크포인트 저장

### 5.2 유지보수성
- 모듈화된 코드 구조
- 명확한 함수/클래스 문서화
- 재현 가능한 실험 (seed 고정)

### 5.3 확장성
- 새 모델 추가 용이
- 필터 규칙 추가/수정 가능
- 평가 벤치마크 확장 가능

---

## 6. 제약사항

### 6.1 기술적 제약
| 제약 | 설명 | 대응 방안 |
|------|------|----------|
| Colab GPU 메모리 | T4: 16GB | 4-bit 양자화, LoRA |
| Colab 시간 제한 | 12시간 | 체크포인트 저장 |
| 모델 크기 | 7B 이하 권장 | 3B 모델 중심 |

### 6.2 비용 제약
| 항목 | 제한 | 비고 |
|------|------|------|
| API 비용 | $0 | API 사용 금지 |
| Colab | 무료 우선 | Pro는 최후 수단 |
| 기타 서비스 | $0 | 오픈소스만 사용 |

### 6.3 시간 제약
- 4주 내 완료
- 발표 자료 준비 시간 포함

---

## 7. 산출물

### 7.1 코드 산출물
```
synthetic-instruction-tuner/
├── src/
│   ├── data_generation/      # Magpie 구현
│   ├── filtering/            # 품질 필터링
│   ├── preference/           # 선호 데이터 생성
│   ├── training/             # SFT, DPO 학습
│   └── evaluation/           # 평가 스크립트
├── notebooks/                # Colab 노트북
├── data/                     # 생성 데이터
├── models/                   # 체크포인트
└── evaluation/               # 평가 결과
```

### 7.2 문서 산출물
| 문서 | 설명 | 형식 |
|------|------|------|
| 요구사항 설명서 | 본 문서 | Markdown |
| 프로젝트 계획서 | 상세 일정 | Markdown |
| 기술 보고서 | 과정 및 결과 | PDF |
| 발표 자료 | 프레젠테이션 | PPT/PDF |

### 7.3 모델 산출물
| 모델 | 설명 | 공개 |
|------|------|------|
| Llama-3.2-3B-SFT-DPO | 미세조정 완료 | Hugging Face Hub |
| Mistral-7B-SFT-DPO | 미세조정 완료 | Hugging Face Hub |
| Qwen2.5-3B-SFT-DPO | 미세조정 완료 | Hugging Face Hub |

### 7.4 데이터 산출물
| 데이터셋 | 규모 | 공개 |
|----------|------|------|
| Synthetic Instructions | 10,000개 | Hugging Face Hub |
| Preference Pairs | 10,000개 | Hugging Face Hub |

---

## 8. 성공 기준

### 8.1 필수 성공 기준
- [ ] 합성 데이터 10,000개 이상 생성
- [ ] 3개 모델 SFT+DPO 완료
- [ ] IFEval, MT-Bench, MMLU 평가 완료
- [ ] Base 대비 성능 향상 확인
- [ ] 보고서 및 발표 자료 완성

### 8.2 추가 성공 기준 (인턴십 어필)
- [ ] AgentBench 평가 완료
- [ ] Hugging Face Hub 공개
- [ ] GitHub 코드 공개
- [ ] 재현 가능한 노트북 제공

### 8.3 성능 목표 (참고)
| 벤치마크 | Base | 목표 (SFT+DPO) |
|----------|------|----------------|
| IFEval | ~30% | 40%+ |
| MT-Bench | ~4.0 | 5.0+ |
| MMLU | ~45% | 45%+ (유지) |

---

## 9. 리스크 관리

### 9.1 식별된 리스크
| 리스크 | 영향도 | 발생확률 | 대응 방안 |
|--------|--------|----------|----------|
| Colab 연결 끊김 | 높음 | 높음 | 체크포인트 자주 저장 |
| GPU OOM | 높음 | 중간 | 배치 크기 축소, 양자화 |
| 모델 수렴 실패 | 높음 | 낮음 | 하이퍼파라미터 조정 |
| 데이터 품질 저하 | 중간 | 중간 | 필터 기준 강화 |
| 시간 부족 | 높음 | 중간 | 모델 수 축소 (3→2) |

### 9.2 비상 계획
- **시간 부족 시**: Mistral-7B 제외, 3B 모델 2개만 진행
- **GPU 부족 시**: Colab Pro 1개월 결제 ($10)
- **성능 미달 시**: 데이터 양 증가 또는 에폭 증가

---

## 10. 변경 이력

| 버전 | 날짜 | 변경 내용 | 작성자 |
|------|------|----------|--------|
| 1.0 | TBD | 초기 작성 | - |

---

*본 문서는 프로젝트 진행 중 수정될 수 있습니다.*
