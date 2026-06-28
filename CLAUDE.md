# Claude Code Instructions for FastConformer Quranic ASR Project

This document provides comprehensive guidance for Claude Code when working with this repository.

---

## Project Overview

**Project:** Progressive Unfreezing for Streaming Arabic Quranic Speech Recognition

**Goal:** Fine-tune NVIDIA's FastConformer model on Quranic Arabic (tarteel-ai/everyayah dataset) using progressive layer unfreezing while maintaining real-time streaming capability.

**Current Status:**
- ✅ Phase 1 Complete (15.60% WER)
- ✅ Phase 2 Complete (14.32% WER)
- ⏳ Phase 3 Pending (Target: 12-13% WER)

**Key Achievement:** Successfully adapted pre-trained model to specialized domain with causal streaming attention.

---

## Repository Structure

```
fast_conformer/
├── quran_fastconformer_finetune.ipynb    # Main 3-phase training notebook (used for Phase 1)
├── phase2_corrected_fixed.ipynb          # Phase 2 standalone notebook
├── phase3_corrected_fixed.ipynb          # Phase 3 standalone notebook (ready to run)
├── restart_phase1_fixed.py               # Phase 1 resume script
├── RESEARCH_PAPER.md                     # Complete research documentation (UPDATE THIS!)
├── CLAUDE.md                             # This file
├── env.py                                # Contains HF_TOKEN, WANDB_KEY (DO NOT COMMIT)
├── quran_asr/                            # All outputs
│   ├── data/
│   │   ├── train_manifest.jsonl          # 167,908 samples
│   │   ├── val_manifest.jsonl            # 20,976 samples
│   │   └── test_manifest.jsonl           # 20,914 samples
│   ├── checkpoints/
│   │   ├── phase1_top3/                  # Phase 1 checkpoints
│   │   │   ├── last.ckpt                 # Step 2000, 15.60% WER
│   │   │   └── phase1_top3_wer0.1560.nemo
│   │   ├── phase2_layers_9_17/           # Phase 2 checkpoints
│   │   │   ├── last.ckpt                 # Step 3000, 14.32% WER
│   │   │   └── phase2_layers_9_17_wer0.1432.nemo
│   │   └── phase3_full_finetune/         # Phase 3 checkpoints (pending)
│   ├── logs/                             # Training logs
│   └── tokenizer/                        # Tokenizer files (base model's 1024 BPE)
└── wandb/                                # Weights & Biases logs
```

---

## Critical Design Decisions

### 1. Tokenizer Vocabulary (CRITICAL!)

**Decision:** Use base model's original 1024-token BPE vocabulary.

**DO NOT:**
- ❌ Train custom tokenizer
- ❌ Swap tokenizer between phases
- ❌ Modify vocabulary size

**Reason:** Custom tokenizer caused catastrophic encoder-decoder mismatch in early experiments (WER stuck at 100%). See RESEARCH_PAPER.md Section 4.1 for details.

**Validation:**
```python
assert asr_model.tokenizer.vocab_size == 1024, "Wrong tokenizer!"
```

### 2. Streaming Configuration (CRITICAL!)

**Decision:** Apply causal attention from Phase 1 start, not post-training.

**Configuration:**
```python
STREAMING_LEFT_CONTEXT  = 128  # ~1.28s lookback
STREAMING_RIGHT_CONTEXT = 0    # Fully causal
STREAMING_CONV_CONTEXT  = "causal"
```

**Always call after loading checkpoint:**
```python
apply_streaming_config(asr_model)
assert asr_model.encoder.att_context_size[1] == 0, "Not causal!"
```

**Reason:** `restore_from()` doesn't reliably preserve attention mode changes. Must explicitly reapply.

### 3. Progressive Unfreezing Strategy

| Phase | Trainable Layers | Frozen Layers | Steps | LR | Trainable % |
|-------|-----------------|---------------|-------|-----|-------------|
| 1 | 15-17, decoder, joint | 0-14, pre-encoder | 2000 | 5e-5 | 21% |
| 2 | 9-17, decoder, joint | 0-8, pre-encoder | 3000 | 1e-4 | 50% |
| 3 | All (0-17, decoder, joint, pre-encoder) | None | 2000 | 1e-5 | 100% |

**DO NOT** change layer freeze boundaries without understanding implications!

### 4. Scheduler Configuration (BUG FIX!)

**MUST include in `update_data_config()`:**
```python
model.cfg.optim.sched.max_steps = max_steps
```

**Reason:** NeMo requires `max_steps` in model config, not just PyTorch Lightning Trainer. Omitting this causes scheduler malfunction.

### 5. Trainer State Reset (BUG FIX!)

**After each phase:**
```python
asr_model._trainer = None
```

**Reason:** Prevents stale trainer references from leaking between phases.

---

## Common Tasks

### Task 1: Resume Training from Checkpoint

**Scenario:** Training stopped mid-phase.

**Solution:**
```python
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# Load checkpoint
RESUME_CKPT = "./quran_asr/checkpoints/phase2_layers_9_17/last.ckpt"
ckpt = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)
current_step = ckpt['global_step']

# Reapply freeze config
freeze_encoder_layers(asr_model, freeze_up_to=9)  # Phase 2 example
update_data_config(asr_model, train_manifest, val_manifest, lr=1e-4, max_steps=3000)

# Build new trainer (no exp_manager to avoid path conflicts)
trainer = pl.Trainer(...)
trainer.fit(asr_model, ckpt_path=RESUME_CKPT)
```

### Task 2: Evaluate Model on Test Set

```python
def evaluate_wer(model, manifest_path, batch_size=8):
    from nemo.collections.asr.metrics.wer import word_error_rate
    model.eval()
    model = model.to(device)

    audio_paths, texts = [], []
    with open(manifest_path) as f:
        for line in f:
            e = json.loads(line)
            audio_paths.append(e["audio_filepath"])
            texts.append(e["text"])

    hypotheses = []
    for i in tqdm(range(0, len(audio_paths), batch_size)):
        batch = audio_paths[i : i + batch_size]
        with torch.no_grad():
            results = model.transcribe(batch, batch_size=batch_size)
        for r in results:
            hypotheses.append(r.text if hasattr(r, "text") else str(r))

    wer = word_error_rate(hypotheses=hypotheses, references=texts[:len(hypotheses)])
    model.train()
    return wer

# Usage
wer_test = evaluate_wer(asr_model, "./quran_asr/data/test_manifest.jsonl")
```

### Task 3: Compare Streaming vs Non-Streaming WER

```python
from omegaconf import open_dict

# Save current streaming config
original_mode = asr_model.encoder.self_attention_model
original_ctx = asr_model.encoder.att_context_size

# Evaluate with streaming (current)
wer_streaming = evaluate_wer(asr_model, val_manifest)
print(f"Streaming WER: {wer_streaming:.4f}")

# Switch to full attention
with open_dict(asr_model.cfg):
    asr_model.change_attention_model(
        self_attention_model="rel_pos",
        att_context_size=[-1, -1]
    )

# Evaluate with full attention
wer_full = evaluate_wer(asr_model, val_manifest)
print(f"Full attention WER: {wer_full:.4f}")
print(f"Streaming penalty: {(wer_streaming/wer_full - 1)*100:.1f}%")

# Restore streaming config
apply_streaming_config(asr_model)
```

### Task 4: Push Checkpoint to HuggingFace

```python
def push_to_hf(model, phase_name, checkpoint_dir, wer):
    nemo_path = f"{checkpoint_dir}/{phase_name}_wer{wer:.4f}.nemo"
    model.save_to(nemo_path)

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=nemo_path,
        path_in_repo=f"{phase_name}/{os.path.basename(nemo_path)}",
        repo_id="mohammed/fastconformer-quran-ar",
        repo_type="model",
        commit_message=f"{phase_name} — WER {wer:.4f}"
    )
```

### Task 5: Update Research Paper

**After completing each phase or experiment:**
```bash
# Update RESEARCH_PAPER.md with new results
# Sections to update:
# - Section 3 (Results) - Add new phase results
# - Section 4 (Analysis) - Add new insights
# - Section 6 (Future Work) - Update based on findings
```

**Template for adding Phase 3 results:**
```markdown
### 3.3 Phase 3: Full Fine-Tuning

**Training Timeline:**
- Start: [DATE], [TIME]
- Completion: [DATE], [TIME]
- Duration: ~[X] hours

**WER Progression:**

| Step | WER | Time | Notes |
|------|-----|------|-------|
| 0 (baseline) | 14.32% | - | Phase 2 final |
| 500 | [X.XX]% | [TIME] | [NOTES] |
| 1000 | [X.XX]% | [TIME] | [NOTES] |
| 1500 | [X.XX]% | [TIME] | [NOTES] |
| 2000 | [X.XX]% | [TIME] | [NOTES] |
| Final (eval) | **[X.XX]%** | [TIME] | Phase 3 complete |

**Key Observations:**
- Total improvement: [X.XX]% → [X.XX]% (↓[X.XX]% absolute, ↓[X.X]% relative)
- [OBSERVATION 1]
- [OBSERVATION 2]
```

---

## Common Issues and Solutions

### Issue 1: "NameError: CKPT_DIR not defined"

**Cause:** Running notebook cells out of order.

**Solution:** Always run Cell 4 (Configuration) first after kernel restart.

### Issue 2: "NotFoundError: No checkpoints found"

**Cause:** Using `exp_manager` with `resume_if_exists=True` looking in wrong directory.

**Solution:** Use manual logger setup (no `exp_manager`) for resume scenarios:
```python
tb_logger = TensorBoardLogger(save_dir=PHASE_CKPT, name="", version="")
wandb_logger = WandbLogger(project=WANDB_PROJECT, name=PHASE, save_dir=PHASE_CKPT)
trainer = pl.Trainer(..., logger=[tb_logger, wandb_logger])
```

### Issue 3: Training WER Stuck at 100%

**Cause:** Likely tokenizer vocabulary mismatch (see Critical Design Decision 1).

**Solution:**
1. Verify vocabulary size: `assert asr_model.tokenizer.vocab_size == 1024`
2. DO NOT swap tokenizers between base model and training
3. If swapped, reload base model and restart from Phase 1

### Issue 4: Streaming Config Not Persisting

**Cause:** `restore_from()` doesn't preserve attention mode changes.

**Solution:** Always call `apply_streaming_config()` after loading checkpoint:
```python
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(checkpoint)
apply_streaming_config(asr_model)  # MUST CALL THIS!
assert asr_model.encoder.att_context_size[1] == 0
```

### Issue 5: CUDA Out of Memory

**Cause:** Batch size too large for 12GB VRAM.

**Solution:** Reduce batch size, increase gradient accumulation:
```python
BATCH_SIZE = 4  # Reduce from 6
GRAD_ACCUM = 12  # Increase from 8
# Effective batch size stays 48
```

---

## Testing and Validation

### Pre-Training Checklist

Before starting any training phase:
- [ ] Config cell (Cell 4) executed
- [ ] Authentication successful (W&B + HuggingFace)
- [ ] Manifests verified (train, val, test exist)
- [ ] GPU available (`torch.cuda.is_available() == True`)
- [ ] Checkpoint paths exist for resume scenarios
- [ ] Streaming config verified (`att_context_size[1] == 0`)
- [ ] Vocabulary size correct (`vocab_size == 1024`)

### Post-Training Validation

After completing a phase:
- [ ] WER improved (or stayed stable)
- [ ] Checkpoint saved (`.ckpt` and `.nemo`)
- [ ] Pushed to HuggingFace
- [ ] Logged to Weights & Biases
- [ ] RESEARCH_PAPER.md updated
- [ ] Trainer state reset (`asr_model._trainer = None`)

---

## Important File Paths

**Configuration:**
```python
BASE_DIR = "./quran_asr"
DATA_DIR = f"{BASE_DIR}/data"
CKPT_DIR = f"{BASE_DIR}/checkpoints"
LOG_DIR = f"{BASE_DIR}/logs"
```

**Manifests:**
```python
train_manifest = "./quran_asr/data/train_manifest.jsonl"
val_manifest = "./quran_asr/data/val_manifest.jsonl"
test_manifest = "./quran_asr/data/test_manifest.jsonl"
```

**Phase Checkpoints:**
```python
PHASE1_CHECKPOINT = "./quran_asr/checkpoints/phase1_top3/last.ckpt"
PHASE2_CHECKPOINT = "./quran_asr/checkpoints/phase2_layers_9_17/last.ckpt"
PHASE3_CHECKPOINT = "./quran_asr/checkpoints/phase3_full_finetune/last.ckpt"  # Pending
```

**Research Documentation:**
```python
RESEARCH_PAPER = "./RESEARCH_PAPER.md"  # UPDATE AFTER EACH EXPERIMENT!
```

---

## Experiment Tracking

### Weights & Biases

**Project:** `quran-fastconformer`

**Runs:**
- `phase1_top3` - Phase 1 training
- `phase1_top3_resume_step1812` - Phase 1 resume
- `phase2_layers_9_17` - Phase 2 training
- `phase3_full_finetune` - Phase 3 training (pending)

**Key Metrics to Track:**
- `val_wer` - Validation Word Error Rate
- `train_loss` - Training loss
- `learning_rate` - LR scheduler progression
- `trainable_params` - Number of trainable parameters

### HuggingFace Hub

**Repository:** `mohammed/fastconformer-quran-ar`

**Published Models:**
- `phase1_top3/phase1_top3_wer0.1560.nemo`
- `phase2_layers_9_17/phase2_layers_9_17_wer0.1432.nemo`
- `phase3_full_finetune/phase3_full_finetune_wer[TBD].nemo` (pending)

---

## Research Paper Maintenance

**CRITICAL:** Update `RESEARCH_PAPER.md` after every significant event:

### Update Triggers:
1. ✅ Phase completion (add results to Section 3)
2. ✅ New experiment/ablation (add to Section 4 Analysis)
3. ✅ Bug discovery (add to Section 2.4 Critical Fixes)
4. ✅ Hypothesis validation (update Section 4 Discussion)
5. ✅ New findings (add to Section 6 Future Work)

### Update Template:

When a user completes Phase 3:
```markdown
1. Update Section 3.3 with Phase 3 results
2. Update Section 3.4 Summary Table
3. Add Phase 3 analysis to Section 4
4. Update Section 6 (Future Work) - remove "Complete Phase 3"
5. Update Abstract if final results differ significantly
6. Update Appendix A with Phase 3 log entries
```

---

## Development Workflow

### Starting a New Phase

1. **Verify previous phase complete:**
   ```bash
   ls -lh ./quran_asr/checkpoints/phase[N]/last.ckpt
   ```

2. **Open phase notebook:**
   ```bash
   jupyter notebook phase[N+1]_corrected_fixed.ipynb
   ```

3. **Run all cells in order**

4. **Monitor W&B dashboard**

5. **Update RESEARCH_PAPER.md when complete**

### Debugging a Failed Training

1. **Check recent log file:**
   ```bash
   tail -100 ./quran_asr/logs/phase[N]_*.log
   ```

2. **Verify checkpoint integrity:**
   ```python
   ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
   print(ckpt.keys())
   print(f"Step: {ckpt['global_step']}")
   ```

3. **Check W&B for anomalies:**
   - Loss spikes
   - WER plateaus
   - LR schedule issues

4. **Resume from last good checkpoint**

### Creating Diagnostic Notebooks

For ad-hoc experiments (e.g., streaming vs non-streaming comparison):

1. **Create new notebook:** `diagnostic_[NAME].ipynb`
2. **Load Phase 2 checkpoint** (most stable)
3. **Run experiments**
4. **Document findings in RESEARCH_PAPER.md Section 4 or 6**
5. **DO NOT commit large notebooks** - extract key code to `.py` scripts

---

## Code Style and Conventions

### Naming Conventions

```python
# Phase names
PHASE = "phase1_top3"           # Phase 1
PHASE = "phase2_layers_9_17"    # Phase 2
PHASE = "phase3_full_finetune"  # Phase 3

# Checkpoint directories
PHASE1_CKPT = f"{CKPT_DIR}/phase1_top3"
PHASE2_CKPT = f"{CKPT_DIR}/phase2_layers_9_17"
PHASE3_CKPT = f"{CKPT_DIR}/phase3_full_finetune"

# Learning rates
PHASE1_LR = 5e-5
PHASE2_LR = 1e-4
PHASE3_LR = 1e-5

# Steps
PHASE1_STEPS = 2000
PHASE2_STEPS = 3000
PHASE3_STEPS = 2000
```

### Logging Best Practices

```python
# Always log key events
logger.info(f"=== Starting {PHASE} ===")
logger.info(f"  Steps: {PHASE_STEPS}")
logger.info(f"  LR: {PHASE_LR}")
logger.info(f"  Trainable params: {trainable:,}")

# Log improvements
logger.info(f"  Phase {N} val WER: {wer:.4f}")
logger.info(f"  Improvement over Phase {N-1}: {improvement:.4f} ({rel_improvement:.1f}% relative)")
```

---

## Contact and Collaboration

**Primary User:** Mohammed (m-yousif)

**HuggingFace:** https://huggingface.co/mohammed
**Weights & Biases:** m-yousif
**Project Repository:** (To be added if GitHub repo created)

---

## Version History

- **v1.0** (2026-06-04): Initial version after Phase 2 completion
- **v1.1** (TBD): After Phase 3 completion
- **v2.0** (TBD): After diagnostic evaluations and streaming penalty analysis

---

## Quick Reference: Critical Commands

```python
# Load model
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

# Apply streaming config
apply_streaming_config(asr_model)

# Verify streaming
assert asr_model.encoder.att_context_size[1] == 0

# Verify vocabulary
assert asr_model.tokenizer.vocab_size == 1024

# Freeze layers (Phase 2 example)
freeze_encoder_layers(asr_model, freeze_up_to=9)

# Update config
update_data_config(asr_model, train_manifest, val_manifest, lr=PHASE_LR, max_steps=PHASE_STEPS)

# Resume training
trainer.fit(asr_model, ckpt_path=CHECKPOINT_PATH)

# Reset trainer
asr_model._trainer = None

# Evaluate
wer = evaluate_wer(asr_model, val_manifest)
```

---

**Last Updated:** June 4, 2026
**Status:** Phase 2 Complete, Phase 3 Ready to Run
**Next Action:** Run Phase 3 or diagnostic streaming evaluation
