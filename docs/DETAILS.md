# Methodology Details

Deep-dive companion to the main [README](../README.md). Everything here is either verifiable against a committed artifact (path given) or explicitly marked as a design rationale. A corrections log at the bottom records claims from earlier README versions that did not survive a source audit.

## 1. Magpie Synthetic Data Generation (notebook 02)

Magpie prompting exploits the chat template of instruct-tuned models: send the template with an *empty* user turn, and the model generates both the instruction and the response.

```python
# Template only - no actual instruction
template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

"""
# The model first completes the "user" turn (an instruction),
# then the assistant turn (the response).
```

- **Generator**: `meta-llama/Llama-3.1-8B-Instruct`, 4-bit NF4 quantization (bitsandbytes), sampling with temperature ~0.9–1.0 (see `config.json` and notebook 02).
- **Output**: 1,500 instruction-response pairs → `data/raw/instructions_raw.json` (committed; `instructions_final_full.json` and `instructions_checkpoint.json` are the same run's full/checkpoint variants).
- **Checkpointing**: every 100 samples, so a Colab disconnect loses at most 100 samples. Reruns resume from the latest checkpoint file.
- **Why it is attractive**: no seed dataset, no proprietary API, no human annotation — the marginal cost of 1,500 samples is GPU time only.

## 2. Quality Filtering (notebook 03, `src/filtering/quality_filter.py`)

Six rule-based checks, each returning pass/fail plus a sub-score. The final score is a weighted sum — weights as implemented in `quality_filter.py`:

| Filter | Weight | Checks |
|---|---|---|
| Length | 0.15 | instruction 3–500 words, response 10–2,000 words |
| Language | 0.10 | English/ASCII ratio heuristics |
| Repetition | 0.20 | repeated phrase detection, unique-token ratio ≥ 0.3 |
| Format | 0.15 | required fields present, structure parseable |
| Toxicity | 0.15 | keyword blacklist + pattern matching, refusal detection |
| Content quality | 0.25 | coherence / relevance / informativeness heuristics |

A sample passes when no filter hard-fails and the weighted score ≥ 0.5 (`min_quality_score` in `config.json`).

**Measured outcome** (`evaluation/results/filtering_stats.json`):

- 1,500 in → **1,258 passed (83.9%)**, 242 failed
- Failure reasons: repeated phrases 156, response too short 76, potentially harmful content 5, refusals 5
- Quality score over passing samples: mean 0.88, median 0.90, min 0.66
- The top 1,000 by score were kept → `data/filtered/instructions_filtered.json` / `sft_data.json`, split 900/100 into `sft_train.json` / `sft_val.json`

Takeaway: for synthetic data at this scale, cheap interpretable filters were sufficient; degeneration (phrase loops) is the dominant failure mode of template-only generation, not toxicity.

## 3. Preference Pair Generation (notebook 04, `src/preference/preference_generator.py`)

The variant actually used is `04_preference_generation_STABLE_OPTIMIZED.ipynb` (earlier attempts are in `notebooks/archive/`; the "stable" rewrite processes samples sequentially after the parallel version produced JSON corruption).

Per filtered instruction:

1. Generate up to 4 response variants with the same generator model at **temperatures 0.6 / 0.8 / 1.0 / 1.2** (plus the original response when available).
2. Score every variant with **`OpenAssistant/reward-model-deberta-v3-large-v2`** (scalar reward).
3. Take best as *chosen*, worst as *rejected*; keep the pair only if **margin ≥ 0.5**.

**Measured outcome** (`data/preference/preference_data.json`, 600 pairs):

- Margin: mean 1.78, min 0.53, max 4.83 — the minimum sitting just above 0.5 confirms the gate was enforced in the actual run
- Mean chosen score 0.07 vs mean rejected score −1.71
- Split 540/60 into `dpo_train.json` / `dpo_val.json`

Note on config drift: `config.json` contains a `preference_generation.response_models` list (Llama-3.2-1B, Mistral-7B, Qwen2.5-3B at temperature 0.9). Notebook 04 does **not** use it — the multi-temperature single-generator strategy above is what ran. The config entry is vestigial from an earlier design.

## 4. Training Setups

All from `config.json` and `models/*/final/training_config.json` (committed).

### LoRA SFT (notebook 05)

- r=8, alpha=16, dropout 0.05, bias none; target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all linear projections in the Llama block)
- 3 epochs, lr 2e-4 cosine decay, batch 12 × grad-accum 2 (effective 24) on A100, BF16, 4-bit quantized base
- **Result**: train loss 0.75, **val loss 0.54**, 114 steps (900 × 3 / 24 ≈ 113 — step count matches the data, a useful sanity check)
- Trainable params: 12,156,928 = 0.67% of the 4-bit-loaded model as counted by PEFT (1,815,620,608 total)

### Prompt Tuning (notebook 05b)

- 20 virtual tokens, random init, 61,440 trainable params (0.003%)
- 3 epochs, lr 3e-4, same data and effective batch as LoRA
- `SFTTrainer` does not support `PromptTuningConfig`, so notebook 05b uses a plain `Trainer` with `DataCollatorForLanguageModeling`
- **Result**: avg train loss 5.22, **val loss 2.98** (best checkpoint; `models/prompt_tuning/checkpoint/checkpoint-114/trainer_state.json` shows the full curve: train loss 10.46 at step 1 → 2.97 at step 110, still far above LoRA's plateau)

### DPO (notebook 06)

- beta=0.1, 1 epoch, lr 5e-5, batch 8 × grad-accum 2 (effective 16), frozen SFT model as reference, both models 4-bit
- **Result**: train loss 0.65, **held-out DPO loss 0.55 (starts at ln 2 = 0.693)**, 34 steps (540 / 16 ≈ 34)
- Single-epoch design: 600 pairs is small; additional epochs risk memorizing specific preferences.

### Efficiency measurements (`evaluation/metrics/*.json`)

| | LoRA | Prompt Tuning | DPO |
|---|---|---|---|
| Peak GPU memory | 5.31 GB | 5.94 GB | 4.70 GB |
| Training wall time | 0.137 h (8.2 min) | 0.313 h (18.8 min) | 0.037 h (2.2 min) |
| Inference speed | 7.70 tok/s | 8.44 tok/s | 8.73 tok/s |

DPO's low memory despite dual-model loading comes from sharing the 4-bit base weights and only duplicating adapters.

### T4 vs A100 settings

The committed configs are the A100 run. For the free T4 tier (16 GB): halve batch sizes and double gradient accumulation (SFT 4×4, DPO 2×8), use FP16 instead of BF16 (no BF16 on T4). Generation (notebook 02) runs fine on T4, just slower.

## 5. Evaluation (notebooks 07–09)

What actually exists, honestly labeled:

- **Notebook 07** ("benchmark evaluation") runs *qualitative probes*, not standard benchmarks: 5 constrained instruction-following tests, 5 knowledge questions, and response-statistics over the probe outputs. Raw model outputs are committed in `evaluation/results/instruction_following_results.json` and `knowledge_test_results.json`.
- **Notebook 08** probes agent-style capabilities (multi-step planning, multi-turn context, feedback adaptation) → `evaluation/results/agent_evaluation_results.json`.
- **Notebook 09** aggregates efficiency metrics into `evaluation/metrics/` and the 7 figures in `evaluation/figures/`.

Response statistics over the 10 probes (`evaluation/results/evaluation_summary.json`):

| Model | Avg length (words) | Avg sentences | Avg unique words |
|---|---|---|---|
| Base (zero-shot) | 115.8 | 9.4 | 44.6 |
| SFT (LoRA) | 137.4 (+19%) | 7.0 (−26%) | 71.8 (+61%) |
| DPO | 139.6 (+21%) | 6.0 (−36%) | 74.4 (+67%) |

Interpretation: fine-tuned models write longer, lexically richer, and denser responses (more words per sentence). With n=10 these are directional indicators, not statistics.

## 6. Data Provenance Table

| Claim in README | Backing artifact |
|---|---|
| 1,500 generated pairs | `data/raw/instructions_raw.json` (len 1,500) |
| 83.9% filter pass rate, failure breakdown | `evaluation/results/filtering_stats.json` |
| 1,000 filtered / 900+100 split | `data/filtered/*.json` (lens 1,000 / 900 / 100) |
| 600 preference pairs, margin mean 1.78 min 0.53 | `data/preference/preference_data.json` |
| Val losses 0.54 / 2.98 / 0.55 (DPO value is preference loss) | `models/{sft,prompt_tuning,dpo}/final/training_config.json` |
| Memory / time / throughput table | `evaluation/metrics/{lora,prompt_tuning,dpo}_metrics.json` |
| Response-statistics deltas | `evaluation/results/evaluation_summary.json` |
| Committed adapters ~47 MB each | `models/sft/final/`, `models/dpo/final/` (safetensors in git) |

## 7. Corrections Log (2026-07 source audit)

Claims from earlier README versions, removed or fixed after reading the code and artifacts:

1. **MMLU/HellaSwag/ARC-Easy/TruthfulQA table (57.5% avg, +9.7 vs zero-shot)** — these exact numbers are the hardcoded `# Sample benchmark data` fallback in `notebooks/09_comparative_analysis.ipynb`, exported into `evaluation/metrics/full_comparison_report.json` when `benchmark_results.json` is absent (it is absent; no notebook runs these benchmarks). Removed from README/CV claims; the numbers are still present inside the committed artifacts (`full_comparison_report.json` and the `Avg Benchmark` column of `comparison_summary.csv`). The `benchmark_comparison.png` and `tradeoff_analysis.png` figures plot the same placeholder data and should not be cited as results.
2. **Training times "LoRA 2.87 h / Prompt Tuning 6.62 h"** — contradicted by the repo's own metrics files (0.14 h / 0.31 h). Fixed to the measured values.
3. **Eval losses "2.96" (PT) and "0.54" (DPO)** — actual artifacts say 2.98 and 0.55. Fixed.
4. **Filter weights code block (1.0/1.5/2.0/...)** — the implemented weights in `quality_filter.py` are 0.15/0.10/0.20/0.15/0.15/0.25. Fixed.
5. **Preference "mean margin 1.2"** — measured 1.78. Fixed.
6. **"271 files, ~589 MB" / "~120 MB in Git"** — git tracks 85 files, ~167 MB. Fixed.
7. **Directory structure** — described `models/lora_adapter/`, `data/raw/magpie_data.json`, `09_results_visualization.ipynb`, 16 docs, and an old absolute local root; actual paths are `models/{sft,dpo,prompt_tuning}/final/`, `instructions_raw.json`, `09_comparative_analysis.ipynb`, 4 docs. Fixed.
8. **Placeholder clone URL (`yourusername`)** — replaced with the real repository URL.
9. **Response-statistics table (287 chars / 82 unique words etc.)** — contradicted by `evaluation_summary.json` (115.8 words / 44.6 unique). Fixed to the artifact values.
10. **LoRA rank/alpha ablation results (r=4: 0.62, r=16: 0.55, alpha sweep)** and **"95% constraint adherence (19/20 tests)"** — no artifacts for these runs exist in the repo; removed rather than kept unverifiable.
11. **90/10 vs 80/20 split** — `config.json` says 0.8/0.2 but the committed splits are 900/100 and 540/60 (90/10). The artifacts are authoritative.

## 8. Practical Notes and Troubleshooting

- **Colab disconnects**: generation and training checkpoint every 100 samples/steps; reruns auto-resume. Keep checkpoints on Drive, not ephemeral storage.
- **OOM**: halve `per_device_train_batch_size` and double `gradient_accumulation_steps`; 4-bit base loading is already on by default.
- **Gated models**: `HF_TOKEN` must belong to an account approved for the Llama models; check with `huggingface-cli whoami`.
- **Prompt tuning with TRL**: `SFTTrainer` rejects `PromptTuningConfig` — use a plain `Trainer` (already done in 05b).
- **Sequential beats parallel for preference generation on Colab**: the archived parallel notebook 04 variants produced corrupted JSON under memory pressure; the STABLE_OPTIMIZED sequential version is slower but completed the full 600 pairs.

## 9. Next Steps

- Run `lm-eval-harness` on the committed SFT/DPO adapters to produce real benchmark numbers (the dependency is already listed; a T4 suffices for 3B 4-bit inference).
- Commit the prompt-tuning adapter (~250 KB) or note its absence in the release; currently blocked by the blanket `*.safetensors` gitignore rule.
- Scale test: the pipeline is size-agnostic — raising `target_raw_samples` in `config.json` is the only required change for a larger corpus.
