# Cross-lingual Part-of-Speech Tagging on Basque: Zero-shot vs. In-language Training

This repository contains the code for a study comparing zero-shot cross-lingual transfer against in-language supervision for part-of-speech (POS) tagging, using English as the source language and Basque as the target language.

## Overview

Multilingual pretrained language models such as XLM-RoBERTa have shown strong zero-shot cross-lingual transfer, but it remains unclear how much of this reflects genuinely shared, linguistic representations versus surface-level similarity between related languages. This project investigates that question by evaluating cross-lingual POS tagging transfer from English to Basque, a language isolate with no genetic relatives, making it a linguistically motivated test for cross-lingual transfer.

Three systems are compared under matched conditions:

- **System A (Zero-shot)** — XLM-RoBERTa fine-tuned only on English (`UD_English-EWT`), evaluated directly on Basque with no exposure to labeled Basque data during training.
- **System B (In-language)** — XLM-RoBERTa fine-tuned directly on Basque (`UD_Basque-BDT`).
- **Frequency-based Baseline** — A frequency-based baseline that tags each word with its most frequent UPOS tag observed in training, evaluated under both zero-shot and in-language conditions.

The better-performing system (System B) is then used to annotate the held-out Basque development set, producing a newly annotated corpus.

## Repository Structure

```
.
├── data_set_prep.py           # Parses CoNLL-U files, defines the UPOS label set and label<->id mappings
├── system_a.py                # Trains and evaluates System A (zero-shot, English-trained)
├── system_b.py                # Trains and evaluates System B (in-language, Basque-trained)
├── baseline.py                # Trains and evaluates the frequency-based baseline (both conditions)
├── new_annotation.py          # Uses the trained System B model to annotate the Basque dev set
├── eu_bdt-ud-dev.system_b_annotated.conllu   # Output: newly annotated Basque corpus
└── README.md
```

## Data

This project uses the Universal Dependencies treebanks, downloaded in CoNLL-U format:

- [UD_English-EWT](https://github.com/UniversalDependencies/UD_English-EWT) — used for training System A
- [UD_Basque-BDT](https://github.com/UniversalDependencies/UD_Basque-BDT) — train/test sets used for System B; dev set used for the final annotation task

> **Note:** Data files are not included in this repository. To reproduce the experiments, download the files below from the treebank repositories above and place them in the project root:
> - `en_ewt-ud-train.conllu`
> - `eu_bdt-ud-train.conllu`
> - `eu_bdt-ud-test.conllu`
> - `eu_bdt-ud-dev.conllu`

## Setup

```bash
pip install transformers datasets seqeval conllu torch accelerate
```

## Usage

If using a virtual environment, activate it first:

```bash
source .venv/bin/activate
```

Run each script from the project root, with the required `.conllu` files present in the same directory.

Train and evaluate System A (zero-shot):
```bash
python system_a.py
```

Train and evaluate System B (in-language):
```bash
python system_b.py
```

Train and evaluate the frequency-based baseline (both conditions):
```bash
python baseline.py
```

Annotate the Basque dev split using the better-performing system:
```bash
python new_annotation.py
```

> Update the model checkpoint path in `new_annotation.py` to point to saved System B checkpoint before running.

## Pipeline

Each XLM-R-based system follows the same four-stage pipeline:

1. **Data cleaning and preparation** — Raw CoNLL-U files are parsed and filtered (multiword tokens and empty nodes removed), producing word-level sequences aligned one-to-one with their UPOS tags.
2. **Tokenisation and label alignment** — Words are split into subwords by the XLM-R tokenizer; each word's gold label is assigned only to its first subword, with remaining subwords and special tokens excluded from the loss.
3. **Training** — The token-classification model is fine-tuned on the resulting token-aligned examples.
4. **Testing** — Predictions are compared against gold labels to compute accuracy, precision, recall, and F1 using `seqeval`.

The frequency-based baseline follows a simplified version of this pipeline, replacing tokenisation and subword alignment with a simple word-to-tag frequency count.

## Results

| System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| System A (Zero-shot) | 0.779 | 0.730 | 0.703 | 0.717 |
| System B (In-language) | 0.954 | 0.937 | 0.939 | 0.938 |
| Baseline (Zero-shot) | 0.416 | 0.491 | 0.186 | 0.269 |
| Baseline (In-language) | 0.860 | 0.827 | 0.795 | 0.810 |

System B was selected to annotate the held-out Basque dev split. As a reliability check, its predictions on the dev split were evaluated against the gold annotations, yielding an F1 of 0.937, nearly identical to its test-set performance (F1 = 0.938).

## Newly Annotated Corpus

The output file `eu_bdt-ud-dev.system_b_annotated.conllu` contains the `UD_Basque-BDT` development set with UPOS tags predicted by System B, in the original CoNLL-U format with all other annotation layers (e.g., dependency relations) preserved unchanged.

## Model and Libraries

- **XLM-RoBERTa** ([Conneau et al., 2020](https://aclanthology.org/2020.acl-main.747/)) via HuggingFace `transformers`
- **seqeval** ([Nakayama, 2018](https://github.com/chakki-works/seqeval)) for evaluation
- **conllu** for CoNLL-U parsing

