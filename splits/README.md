# Dataset Splits

This directory contains the training and validation data subsets used in the experiments reported in the paper.

## Recoder

The Recoder dataset is used for training and validating the repair-pattern experts and the Full LoRA baseline.

Original source:

- Recoder：https://github.com/pkuzqh/Recoder

Files:

```
recoder/
├── src-train.jsonl
├── tgt-train.jsonl
├── src-valid.jsonl
└── tgt-valid.jsonl
```

## TRANSFER

The TRANSFER dataset is used for training and validating the gating component in the standard X-LoRA setting.

Original source:

- TRANSFER：https://github.com/mxx1219/TRANSFER

Files:

```
transfer/
├── src-train.jsonl
├── tgt-train.jsonl
├── src-valid.jsonl
└── tgt-valid.jsonl
```