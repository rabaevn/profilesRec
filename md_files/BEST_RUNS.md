# Best Runs

## Phase 2 — Beauty (all text flags)

**Test recall@10: 0.0991 | ndcg@10: 0.0491**

| metric       | val (best ckpt) | test   |
|--------------|-----------------|--------|
| recall@5     | —               | 0.0623 |
| ndcg@5       | —               | 0.0373 |
| recall@10    | 0.1056          | 0.0991 |
| ndcg@10      | 0.0518          | 0.0491 |

Best val checkpoint reached at epoch 3 of 5.

### Hyperparameters

```
--model-name              Qwen/Qwen3-Embedding-0.6B
--dataset-name            Beauty
--num-train-epochs        5
--learning-rate           1e-4
--train-batch-size        256
--eval-batch-size         8
--warmup-ratio            0.1
--weight-decay            0.01
--gradient-accumulation-steps  1
--fp16
--enrich-text-input            # --history-sep + --history-time-text + --history-rating-text + --history-pos-marker
--best-metric             recall@10
```

### Phase 2 Flag Ablation (Beauty, same hyperparams)

| flags           | recall@10 | ndcg@10 |
|-----------------|-----------|---------|
| all 4 (best)    | 0.0991    | 0.0491  |
| time only       | 0.0984    | 0.0484  |
| rating only     | 0.0976    | 0.0480  |
| pos only        | 0.0969    | 0.0480  |
| sep only        | 0.0963    | 0.0476  |

All flags together beat any individual flag — synergistic effect.

### Artifacts
- Local metrics: `outputs/phase2_qwen3_Beauty_all_flags/`
- Log: `logs/phase2_qwen3_Beauty_all_flags_gpu0.log`
- GFS ablations: `/gfs/shared/shbeni/seqrec/outputs/phase2_qwen3_Beauty_*/`
