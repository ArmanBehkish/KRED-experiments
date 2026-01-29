# KRED: Knowledge-Aware Document Representation for News Recommendations

> **An extended study of the KRED framework** — ablation analysis, NLU model substitution, and fine-grained classification experiments on the MIND dataset.

This repository contains the implementation and extended experiments on the [KRED paper](https://arxiv.org/abs/1910.11494) (Liu et al., ACM RecSys 2020). The work was carried out as a Deep NLP course project at Politecnico di Torino.

| Resource | Link |
|---|---|
| Original paper | [KRED: Knowledge-Aware Document Representation for News Recommendations](https://arxiv.org/abs/1910.11494) |
| Final report (PDF) | [docs/Final_Report_KRED_Team_Bax.pdf](docs/Final_Report_KRED_Team_Bax.pdf) |
| Presentation (PPTX) | [docs/KRED Presentation last version.pptx](<docs/KRED Presentation last version.pptx>) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [KRED Architecture](#kred-architecture)
3. [NLP and LLM Concepts Covered](#nlp-and-llm-concepts-covered)
4. [Extensions and Experiments](#extensions-and-experiments)
5. [Results](#results)
6. [Dataset](#dataset)
7. [Repository Structure](#repository-structure)
8. [Getting Started](#getting-started)
9. [Configuration](#configuration)
10. [References](#references)

---

## Project Overview

News recommendation poses unique challenges compared to traditional recommendation systems: articles are highly time-sensitive, collaborative filtering alone is insufficient, and content understanding is critical. **KRED** addresses this by fusing knowledge-graph information with neural document representations to produce **Knowledge-enhanced Document Vectors (KDV)** that can drive multiple downstream tasks.

This project reproduces the KRED pipeline on the **MIND-small** dataset and extends it with three lines of investigation:

1. **NLU model substitution** — replacing BERT with RoBERTa and MiniLM to measure the impact of the backbone language model.
2. **Context embedding ablation study** — isolating position, frequency, and type encodings to quantify their individual contributions.
3. **Fine-grained classification** — scaling the category head from 17 general topics to 263 sub-categories to test robustness.

---

## KRED Architecture

![KRED Framework](./framework.PNG)

KRED produces a knowledge-enhanced document vector through three layers:

### 1. Entity Representation Layer
- Extracts named entities (people, places, events) from news articles and links them to a **knowledge graph** (Wikidata).
- Uses **TransE** to learn low-dimensional embeddings for entities and relations via *(head, relation, tail)* triplets.
- Applies a **Knowledge Graph Attention Network (KGAT)** to propagate and aggregate information from one-hop entity neighbours, weighting them with an attention mechanism.

### 2. Context Embedding Layer
- Enriches each entity vector with dynamic contextual signals:
  - **Position encoding** — whether the entity appears in the title or the body.
  - **Frequency encoding** — occurrence count of the entity (capped at 20).
  - **Type (category) encoding** — the entity's category within the knowledge graph.

### 3. Information Distillation Layer
- An **attention layer** that fuses all entity representations using the original document vector as the query and entity vectors as keys/values.
- The output is concatenated with the original document vector and passed through a dense layer to produce the final **Knowledge-enhanced Document Vector (KDV)**.

The KDV is then fed into task-specific heads for user-to-item ranking, item-to-item similarity, category classification, or popularity prediction.

---

## NLP and LLM Concepts Covered

This project touches on a broad range of foundational and modern NLP / language-model techniques:

### Core NLP Concepts
| Concept | Role in This Project |
|---|---|
| **Named Entity Recognition (NER)** | Extracting entities from news articles to link to the knowledge graph |
| **Knowledge Graphs & Knowledge Graph Embeddings** | Wikidata graph with TransE-based entity/relation embeddings |
| **Text Classification** | Category classification head (17-class and 263-class settings) |
| **Attention Mechanisms** | KGAT for entity neighbour aggregation; information distillation layer for entity-document fusion |
| **Multi-Task Learning** | Joint training across recommendation, classification, and popularity prediction objectives |
| **Ablation Study Methodology** | Systematic removal of context embedding components to measure contribution |
| **Evaluation Metrics** | Accuracy, macro F1 score, AUC, NDCG@10 |

### Transformer-Based Language Models (LLMs)
| Model | Details |
|---|---|
| **BERT** (baseline) | Bidirectional encoder used in the original KRED paper for document embeddings (768-dim) |
| **RoBERTa** | Robustly optimised BERT variant; trained on larger data with improved pre-training strategy; fine-tuned with sentence-similarity objective |
| **MiniLM** | Distilled transformer with fewer parameters; tests the trade-off between model size and downstream performance |
| **Sentence-Transformers** | Framework used to generate fixed-size sentence embeddings from the above models (`distilbert-base-nli-stsb-mean-tokens` in the default pipeline) |

### Representation Learning
| Technique | Application |
|---|---|
| **TransE** | Translational embedding model for knowledge-graph triples |
| **KGAT (Knowledge Graph Attention Network)** | Graph neural network with attention over heterogeneous entity-relation edges |
| **Positional / Frequency / Type Encoding** | Contextual feature injection into entity embeddings |
| **Concatenation + Dense Fusion** | Combining knowledge-enhanced embeddings with raw document vectors |

---

## Extensions and Experiments

### Extension 1 — NLU Model Substitution

The original KRED implementation uses BERT to produce document vectors. We replaced it with two alternative models to study the sensitivity of the framework to the quality of its backbone encoder:

- **RoBERTa** — a robustly optimised BERT pre-training approach (Liu et al., 2019), fine-tuned with a sentence-similarity objective for semantically meaningful representations.
- **MiniLM** — a lightweight distilled transformer (Wang et al., 2020) with significantly fewer parameters, testing the speed-vs-quality trade-off.

**Key finding:** The backbone NLU model is the single most impactful component. Upgrading to RoBERTa yielded **+4.37% accuracy** and a large F1 improvement (+0.16), while downgrading to MiniLM caused a **-5.73% accuracy** drop.

### Extension 2 — Fine-Grained Sub-Category Classification

The default task classifies news into **17 general topics**. We extended the classification head to predict **263 sub-categories** (e.g., distinguishing "NBA" from "Soccer" within "Sports") to assess model robustness at finer granularity.

**Key finding:** Performance dropped sharply (accuracy 34.68%, F1 0.02), revealing that the current entity-based knowledge enrichment is insufficient for fine-grained topic discrimination without additional architectural changes.

### Extension 3 — Context Embedding Ablation Study

We systematically removed one component at a time from the context embedding layer:

| Removed Component | Effect |
|---|---|
| **Type encoding** | Accuracy dropped by **-0.70%** — the most important context signal |
| **Position encoding** | Accuracy slightly *increased* by **+0.29%** — position may introduce noise |
| **Frequency encoding** | Negligible change (**+0.03%**) — frequency contributes minimally |

**Key finding:** Type encoding is the most valuable contextual feature; position encoding can even hurt performance, suggesting the title-vs-body distinction is less informative for category classification.

---

## Results

### NLU Model Comparison

| Setting | Accuracy | F1 Score | Gain |
|---|---|---|---|
| Baseline (BERT) | 70.81% | 0.35 | — |
| **RoBERTa** | **75.18%** | **0.51** | **+4.37** |
| MiniLM | 65.08% | 0.22 | -5.73 |

### General vs. Sub-Category Classification

| Setting | Accuracy | F1 Score | Gain |
|---|---|---|---|
| 17 Classes (Baseline) | 70.81% | 0.35 | — |
| 263 Classes | 34.68% | 0.02 | -36.13 |

### Context Embedding Ablation

| Setting | Accuracy | F1 Score | Gain |
|---|---|---|---|
| Baseline (all components) | 70.81% | 0.35 | — |
| No Type | 70.11% | 0.34 | -0.70 |
| No Position | 71.10% | 0.36 | +0.29 |
| No Frequency | 70.84% | 0.35 | +0.03 |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Batch size | 64 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning rate | 2e-5 |
| Weight decay | 1e-6 |

---

## Dataset

The project uses the [MIND (Microsoft News Dataset)](https://msnews.github.io) — a large-scale benchmark for news recommendation research.

- **MIND-small**: 50,000 users sampled from users with at least 5 news clicks over 6 weeks.
- **Contents**: impression logs (click history + displayed news), news metadata (category, sub-category, title, entities), and Wikidata knowledge-graph embeddings (100-D entity and relation vectors).
- **Note**: The news body text is not available in the open dataset; only titles and abstracts are used.

---

## Repository Structure

```
KRED/
├── main.py                   # Entry point — training and evaluation
├── train_test.py             # Training loop, testing, dataset classes
├── parse_config.py           # YAML configuration parser
├── config.yaml               # Hyperparameters and data paths
├── kred_example.ipynb        # Quick-start Jupyter notebook
├── framework.PNG             # Architecture diagram
├── relation_embedding.vec    # Pre-trained relation embeddings
│
├── model/
│   ├── KRED.py               # Main KRED model (entity fusion + task heads)
│   ├── News_embedding.py     # Knowledge-enriched news representation
│   ├── User_modeling.py      # User history attention model
│   └── KGAT.py               # Knowledge Graph Attention Network
│
├── trainer/
│   └── trainer.py            # Epoch-based training with early stopping
│
├── base/
│   ├── base_model.py         # Abstract base model
│   ├── base_trainer.py       # Abstract base trainer
│   └── base_data_loader.py   # Base data loader
│
├── utils/
│   ├── util.py               # Data loading, embedding extraction, MIND utilities
│   ├── metrics.py            # AUC, NDCG, accuracy, F1 evaluation
│   └── pytorchtools.py       # Early stopping implementation
│
├── logger/
│   ├── logger.py             # Logging setup
│   └── logger_config.json    # Log configuration (console + rotating file)
│
├── docs/
│   ├── Final_Report_KRED_Team_Bax.pdf    # Full project report
│   └── KRED Presentation last version.pptx  # Project presentation slides
│
└── out/saved/                # Model checkpoints and training logs
```

---

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch 1.4+
- CUDA-compatible GPU (recommended)

### Installation

```bash
pip install torch numpy scikit-learn scipy sentence-transformers pyyaml
```

### Download the MIND Dataset

The dataset will be automatically downloaded on the first run, or you can manually place it under `./data/`.

### Train the Model

```bash
# Default: single-task category classification
python main.py

# Or explore interactively via the notebook
jupyter notebook kred_example.ipynb
```

### Configuration

Edit `config.yaml` to change the task, hyperparameters, or data paths:

```yaml
trainer:
  training_type: "single_task"     # or "multi_task"
  task: "vert_classify"            # user2item | item2item | vert_classify | pop_predict
  epochs: 100
  early_stop: 3
```

---

## References

1. D. Liu, J. Lian, S. Wang, Y. Qiao, J.-H. Chen, G. Sun, and X. Xie, **"KRED: Knowledge-aware document representation for news recommendations,"** in *Fourteenth ACM Conference on Recommender Systems*, ACM, 2020.
2. A. Iana, M. Alam, and H. Paulheim, **"A survey on knowledge-aware news recommender systems,"** *Semantic Web*, 2022.
3. H. Wang, F. Zhang, X. Xie, and M. Guo, **"DKN: Deep knowledge-aware network for news recommendation,"** in *Proceedings of the 2018 World Wide Web Conference*, 2018.
4. A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko, **"Translating embeddings for modeling multi-relational data,"** *Advances in Neural Information Processing Systems*, vol. 26, 2013.
5. F. Wu et al., **"MIND: A large-scale dataset for news recommendation,"** in *Proceedings of the 58th ACL*, 2020.
6. X. Wang, X. He, Y. Cao, M. Liu, and T.-S. Chua, **"KGAT: Knowledge graph attention network for recommendation,"** in *Proceedings of the 25th ACM SIGKDD*, 2019.
7. Y. Liu et al., **"RoBERTa: A robustly optimized BERT pretraining approach,"** *arXiv:1907.11692*, 2019.
8. W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, **"MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,"** *Advances in Neural Information Processing Systems*, vol. 33, 2020.
