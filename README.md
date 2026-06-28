# FastConformer Quran ASR Training

This directory contains all notebooks and logs for fine-tuning the FastConformer model on Quranic Arabic.

## Directory Structure

```
fast_conformer/
├── README.md                           # This file
├── quran_fastconformer_finetune.ipynb # Original full training notebook (Phases 1-3)
├── phase2_training.ipynb              # Phase 2 standalone notebook (NEW)
└── log_files/                         # W&B exported logs
    └── wandb_export_*.csv
```

## Training Progress

### ✅ Phase 1 - Complete
- **Status**: ✓ Completed successfully
- **Notebook**: `quran_fastconformer_finetune.ipynb` (cells 1-20)
- **Duration**: ~2 hours (2000 steps)
- **Results**:
  - Validation WER: **32.21%**
  - Trainable params: 41.5% (top 3 encoder layers + decoder)
  - Checkpoint: `../quran_asr/checkpoints/phase1_top3/phase1_top3_wer0.3221.nemo`
- **HuggingFace**: Pushed to [mohammed/fastconformer-quran-ar](https://huggingface.co/mohammed/fastconformer-quran-ar)

### 🔄 Phase 2 - Ready to Run
- **Status**: Ready to start
- **Notebook**: `phase2_training.ipynb` ⬅️ **USE THIS**
- **Expected Duration**: ~3-4 hours (3000 steps)
- **Configuration**:
  - Frozen layers: 0-8
  - Trainable layers: 9-17 + decoder + joint
  - Trainable params: 48.8%
  - Learning rate: 1e-4
- **Expected Results**:
  - Target WER: **20-25%** (down from 32%)
  - Checkpoint will be saved to `../quran_asr/checkpoints/phase2_upper_half/`

### ⏳ Phase 3 - Pending
- **Status**: Not started (requires Phase 2 completion)
- **Configuration**:
  - Unfreeze ALL layers
  - Learning rate: 1e-5
  - Steps: 2000
- **Expected Results**:
  - Target WER: **15-20%**

## How to Run Phase 2

### Quick Start

1. Open `phase2_training.ipynb` in Jupyter
2. Run all cells sequentially (Shift+Enter or Cell > Run All)
3. Monitor progress in the notebook and on [W&B](https://wandb.ai/m-yousif-kalamtech/quran-fastconformer)

### Step-by-Step

1. **Setup** (Cells 1-3)
   - Imports and configuration
   - Verifies CUDA availability

2. **Helper Functions** (Cell 4)
   - Defines all utility functions

3. **Authentication** (Cell 5)
   - Logs into W&B

4. **Load Checkpoint** (Cell 6)
   - Loads Phase 1 checkpoint

5. **Configure Training** (Cells 7-9)
   - Unfreezes layers 9-17
   - Sets up trainer and callbacks

6. **Train** (Cell 10) ⏰ **~3-4 hours**
   - Runs 3000 training steps
   - Checkpoints every 500 steps

7. **Evaluate & Save** (Cells 11-12)
   - Evaluates on validation set
   - Pushes to HuggingFace

## Monitoring

### W&B Dashboard
- URL: https://wandb.ai/m-yousif-kalamtech/quran-fastconformer
- Metrics: Loss, WER, Learning Rate, GPU utilization

### GPU Monitoring
Run in terminal:
```bash
watch -n 1 nvidia-smi
```

Expected:
- GPU Utilization: 90-100%
- Memory Usage: ~10-11 GB / 12 GB
- Power: ~250W (varies by GPU)

### Local Logs
```bash
tail -f ../quran_asr/logs/phase2_*.log
```

## File Paths

All paths are relative to the notebook location (`fast_conformer/`):

| Resource | Path |
|----------|------|
| Data manifests | `../quran_asr/data/` |
| Tokenizer | `../quran_asr/tokenizer/` |
| Phase 1 checkpoint | `../quran_asr/checkpoints/phase1_top3/` |
| Phase 2 output | `../quran_asr/checkpoints/phase2_upper_half/` |
| Logs | `../quran_asr/logs/` |

## Troubleshooting

### CUDA Out of Memory
If you get OOM errors (unlikely with same settings as Phase 1):

Edit `phase2_training.ipynb`, cell 2:
```python
BATCH_SIZE = 4  # Reduce from 6
GRAD_ACCUM = 12  # Increase from 8 (keeps effective batch=48)
```

### Checkpoint Not Found
Verify Phase 1 checkpoint exists:
```bash
ls -lh ../quran_asr/checkpoints/phase1_top3/phase1_top3_wer0.3221.nemo
```

If missing, you need to complete Phase 1 first.

### Training Stuck
Check if GPU is actually being used:
```bash
nvidia-smi
```

If GPU utilization is 0%, check:
1. CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
2. NeMo is using GPU: Check notebook cell 1 output

## After Phase 2

Once Phase 2 completes successfully:

1. **Review Results**
   - Check final WER on W&B
   - Expected: 20-25% (improvement of ~7-12% absolute)

2. **Create Phase 3 Notebook** (optional)
   - Similar structure to Phase 2
   - Load Phase 2 checkpoint
   - Unfreeze all layers
   - Train with LR=1e-5 for 2000 steps

3. **Final Evaluation**
   - Test on held-out test set
   - Verify streaming performance
   - Benchmark inference speed

## Resources

- **W&B Project**: https://wandb.ai/m-yousif-kalamtech/quran-fastconformer
- **HuggingFace Repo**: https://huggingface.co/mohammed/fastconformer-quran-ar
- **Base Model**: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
- **Dataset**: tarteel-ai/everyayah (167k+ Quranic recitations)

## Contact

For issues or questions, check:
1. W&B logs for detailed metrics
2. Local logs in `../quran_asr/logs/`
3. NeMo error logs in checkpoint directories
