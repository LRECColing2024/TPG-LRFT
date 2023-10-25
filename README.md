# TPG-LRFT

This account is an anonymous account for LREC-Coling 2024.

This code is for paper "Improving Low-Resource Keyphrase Generation through Unsupervised Title Phrase Generation"

## Requirements    
- transformers
- pytorch
- pytorch lightning
- nltk
- tqdm

## Runing
### Constructing Pseudo Keyphrases Label
```shell
python data/mining_phrases_in_titles.py
python data/construct_pseudo_keyphrases.py
```

### Title Phrase Generation (TPG) pre-training
```shell
python train.py \
    --stage TPG \
    --output-dir path/to/save/model \
    --model-path facebook/bart-base \
    --train-path data/kp20k_TPG_train.jsonl \
    --valid-path data/kp20k_TPG_valid.jsonl \
    --batch-size-train 32 \
    --batch-size-valid 16 \
    --max-learning-rate 2e-4 \
    --gpus 0 
```

### Low-Resource Fine-Tuning (LRFT)
```shell
python train.py \
    --stage LRFT \
    --output-dir path/to/save/model \
    --model-path tpg/pretrained/model \
    --train-path data/low-resource_kp20k/kp20k_low-resource_5000_train.jsonl \
    --valid-path data/low-resource_kp20k/kp20k_low-resource_valid.jsonl \
    --batch-size-train 16 \
    --batch-size-valid 8 \
    --max-learning-rate 1e-5 \
    --gpus 0 
```
