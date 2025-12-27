# Colab Pro A100 최적화 플랜 - Synthetic-Instruction-Tuner

## 목표
Google Colab Pro의 A100 GPU (40GB VRAM)에 맞춰 notebooks를 최적화하여:
- 학습 속도 2-3배 향상
- 전체 파이프라인 실행 시간 단축 (33-43시간 → 13-20시간)
- 1,500 샘플 유지 (스케일업 없음)
- 단계별 실행 방식 유지

## 핵심 변경사항

### 1. 배치 사이즈 증가
- **SFT**: 4 → 12 (3배)
- **DPO**: 2 → 8 (4배)
- **Gradient Accumulation**: 4/8 → 2 (감소)

### 2. 체크포인트 간격 조정
- **데이터 생성**: 20 → 100 (5배 감소, 디스크 I/O 최적화)
- **SFT 학습**: 500 → 200 steps
- **DPO 학습**: 200 → 100 steps

### 3. 한국어 설명 추가
- A100 최적화 근거
- 메모리 사용량 예측
- 학습 시간 비교 (T4 vs A100)

---

## 수정할 파일 (우선순위 순)

### ⚠️ CRITICAL: config.json
**파일**: `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\config.json`

#### 변경 1: 체크포인트 간격 (Line 11)
```json
"checkpoint_interval": 100,  // 20 → 100
```

#### 변경 2: Training 섹션 재구조화 (Lines 23-52)
**현재 중첩 구조를 평면 구조로 변경:**
```json
"training": {
  "sft_epochs": 3,
  "sft_batch_size": 12,           // 4 → 12 (3배)
  "sft_learning_rate": 2e-4,
  "dpo_epochs": 1,
  "dpo_batch_size": 8,            // 2 → 8 (4배)
  "dpo_learning_rate": 5e-5,
  "dpo_beta": 0.1,
  "gradient_accumulation_steps": 2,  // 공통 2
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05
}
```

#### 변경 3: LoRA 섹션 삭제 (Lines 54-68)
**전체 섹션 제거** (training 섹션에 통합됨)

---

### HIGH: 05_sft_training.ipynb
**파일**: `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\05_sft_training.ipynb`

#### 변경 1: 헤더 업데이트 (Cell 0)
```markdown
**Training settings** (Colab Pro A100 최적화):
- Batch size: 12 (A100 40GB VRAM 활용, T4 대비 3배 증가)
- Gradient accumulation: 2 (배치 크기 증가로 감소)

**Expected runtime**:
- T4: 6-10 hours
- A100: 2-4 hours (2-3배 빠름)

**A100 최적화 포인트**:
- 더 큰 배치 사이즈로 학습 속도 향상
- 40GB VRAM으로 메모리 효율적인 학습
```

#### 변경 2: 최적화 설명 셀 추가 (Cell 24 이전)
**새 Markdown 셀 삽입:**
```markdown
## A100 최적화 설정

- **Batch size: 12** (T4의 4에서 3배 증가)
  - 40GB VRAM 활용하여 더 큰 배치 처리
  - 메모리 사용량: ~20-25GB / 40GB (안전)

- **Gradient accumulation: 2**
  - Effective batch size: 12 × 2 = 24

- **학습 효과**: 전체 학습 시간 2-3배 단축
```

#### 변경 3: Checkpoint 설정 (Cell 25, ~line 368)
```python
save_steps=200,  # 500 → 200
```

#### 변경 4: Print 문 업데이트 (Cell 25, ~line 383)
```python
print(f"  Batch size: {training_args.per_device_train_batch_size} (T4: 4 → A100: 12, 3배 증가)")
print(f"\n💡 A100 40GB VRAM 활용 → 학습 속도 2-3배 향상 예상")
```

---

### HIGH: 06_dpo_training.ipynb
**파일**: `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\06_dpo_training.ipynb`

#### 변경 1: 헤더 업데이트 (Cell 0)
```markdown
**DPO settings** (Colab Pro A100 최적화):
- Batch size: 8 (A100 40GB VRAM 활용, T4 대비 4배 증가)
- Gradient accumulation: 2

**Expected runtime**:
- T4: 4-6 hours
- A100: 1-2 hours (3-4배 빠름)

**A100 최적화 포인트**:
- DPO는 두 모델 동시 로드 → 매우 메모리 집약적
- A100 40GB로 배치 크기 대폭 증가 (2→8, 4배)
- 메모리 사용량: ~30-35GB / 40GB
```

#### 변경 2: 최적화 설명 셀 추가 (Cell 20 이전)
**새 Markdown 셀:**
```markdown
## A100 최적화 설정

- **Batch size: 8** (T4의 2에서 4배 증가)
  - DPO는 policy + reference model 동시 로드
  - A100 40GB로 배치 8 가능

- **학습 효과**: 전체 학습 시간 3-4배 단축
```

#### 변경 3: Gradient Accumulation 수정 (Cell 20, line 278)
```python
gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
# * 2 제거 (A100: 배치 크기 증가로 불필요)
```

#### 변경 4: Checkpoint 설정 (Cell 20, ~line 293)
```python
save_steps=100,  # 200 → 100
```

#### 변경 5: Print 문 업데이트 (Cell 20, ~line 303)
```python
print(f"  Batch size: {training_args.per_device_train_batch_size} (T4: 2 → A100: 8, 4배 증가)")
print(f"\n💡 A100 40GB VRAM으로 reference model과 함께 큰 배치 사용 가능")
```

---

### MEDIUM: 02_magpie_generation.ipynb
**파일**: `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\02_magpie_generation.ipynb`

#### 변경 1: 헤더 업데이트 (Cell 0)
```markdown
**Expected runtime**:
- T4: 16-17h → 3일 분할 (12시간 제한)
- A100: 6-8h → 한 번에 완료 (런타임 제한 없음)

**Checkpoint Strategy**:
- T4: 500 samples/day, checkpoint every 20
- A100: 1,500 samples 연속, checkpoint every 100

**Tip (A100)**: 한 세션에서 전체 완료 가능
```

#### 변경 2: Checkpoint 메시지 (Cell 17)
```python
if CHECKPOINT_INTERVAL >= 100:
    print(f"   ✓ A100 최적화: 체크포인트 간격 증가로 디스크 I/O 감소")
```

#### 변경 3: Checkpoint 로깅 개선 (Cell 20)
```python
if len(generated_data) % CHECKPOINT_INTERVAL == 0:
    save_checkpoint(generated_data, CHECKPOINT_PATH)
    progress_pct = (len(generated_data) / TARGET_SAMPLES) * 100
    print(f"✓ Checkpoint: {len(generated_data)}/{TARGET_SAMPLES} ({progress_pct:.1f}%)")
```

---

### MEDIUM: 04_preference_generation.ipynb
**파일**: `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\04_preference_generation.ipynb`

#### 변경 1: 헤더 업데이트 (Cell 0)
```markdown
**Expected runtime**:
- T4: 4-6 hours
- A100: 2-3 hours

**Tip (A100)**: 한 세션에서 600 pairs 완료 가능
```

#### 변경 2: Checkpoint 노트 추가 (Cell 22)
```python
print(f"   💡 적절한 간격: reward model 스코어링 상대적으로 빠름")
```

---

## 실행 체크리스트

### Phase 1: 설정 파일 수정 (필수)
- [ ] **백업**: `config.json` 원본 저장
- [ ] `config.json` 수정:
  - [ ] `checkpoint_interval`: 20 → 100
  - [ ] `training` 섹션 재구조화
  - [ ] `sft_batch_size`: 12 추가
  - [ ] `dpo_batch_size`: 8 추가
  - [ ] `gradient_accumulation_steps`: 2 추가
  - [ ] `lora` 섹션 삭제
- [ ] **테스트**: `01_setup.ipynb` 실행하여 config 로드 확인

### Phase 2: 학습 노트북 수정 (높은 우선순위)
- [ ] `05_sft_training.ipynb` 수정:
  - [ ] 헤더 업데이트
  - [ ] A100 설명 셀 추가
  - [ ] `save_steps` 변경
  - [ ] Print 문 업데이트
- [ ] **테스트**: 1 epoch 실행하여 batch=12 동작 확인

- [ ] `06_dpo_training.ipynb` 수정:
  - [ ] 헤더 업데이트
  - [ ] A100 설명 셀 추가
  - [ ] Gradient accumulation 수정
  - [ ] `save_steps` 변경
  - [ ] Print 문 업데이트
- [ ] **테스트**: 100 steps 실행하여 batch=8 동작 확인

### Phase 3: 데이터 생성 노트북 수정 (중간 우선순위)
- [ ] `02_magpie_generation.ipynb` 수정
- [ ] `04_preference_generation.ipynb` 수정

### Phase 4: 전체 검증
- [ ] 10% 서브셋(150 샘플)으로 전체 파이프라인 테스트
- [ ] GPU 사용량 모니터링 (DPO: ~30-35GB 예상)
- [ ] 전체 1,500 샘플 파이프라인 실행

---

## 예상 성능 개선

| 단계 | T4 시간 | A100 시간 | 배속 |
|------|---------|-----------|------|
| 데이터 생성 | 16-17h | 6-8h | 2-2.5x |
| SFT 학습 | 6-10h | 2-4h | 2.5-3x |
| DPO 학습 | 4-6h | 1-2h | 3-4x |
| **전체** | **33-43h** | **13-20h** | **2.5-3x** |

---

## 위험 관리

### OOM 발생 시 대응
**증상**: CUDA out of memory 에러

**해결책**:
1. **SFT OOM**: `sft_batch_size`를 10 또는 8로 감소
2. **DPO OOM**: `dpo_batch_size`를 6 또는 4로 감소
3. `gradient_accumulation_steps`를 4로 증가

**보수적 설정값**:
```json
"sft_batch_size": 8,
"dpo_batch_size": 4,
"gradient_accumulation_steps": 4
```

---

## 주요 파일 경로

1. `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\config.json` ⚠️
2. `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\05_sft_training.ipynb`
3. `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\06_dpo_training.ipynb`
4. `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\02_magpie_generation.ipynb`
5. `D:\Study\Github\TSP\LLM\Synthetic-Instruction-Tuner\notebooks\04_preference_generation.ipynb`
