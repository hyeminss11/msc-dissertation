# msc-dissertation
**Evaluating Authorship Verification Robustness Under Domain Shift and LLM-Based Rewriting**

## Overview
This repository contains the code and experimental framework for my MSc dissertation at the University of Sheffield. The project investigates the robustness of transformer-based authorship verification (AV) models under challenging real-world conditions: domain shift (e.g., news articles vs. tweets) and adversarial rewriting using large language models (LLMs).

## Research Questions
1. **Domain Shift**: Can authorship verification models reliably detect stylistic consistency across different genres when no adversarial rewriting is applied?
2. **Adversarial Robustness**: How robust are these models to LLM-based adversarial rewriting (style obfuscation and impersonation) in same-domain texts?
3. **Combined Challenge**: How do AV models perform when domain shift and adversarial attacks are combined?

## Key Findings
- **DistilBERT** showed superior robustness across all scenarios despite being the smallest model
- **Domain shift impact**: DistilBERT maintained stability (1% drop), while RoBERTa showed catastrophic failure (18% drop)
- **Adversarial attacks**: Impersonation attacks reduced all models to near-random performance (ROC-AUC < 0.56)
- **Combined challenges**: When domain shift and impersonation were combined, performance approached random guessing

## External Dataset: CrossNews

This project uses the [CrossNews](https://github.com/mamarcus64/CrossNews) dataset as a Git submodule for experiments related to authorship verification and threat text analysis.

- Repository: [`external/CrossNews`](external/CrossNews)
- Description: A cross-genre authorship verification and attribution benchmark (news articles and tweets).

> **Citation**:  
> M. Ma, “CROSSNEWS: A Cross-Genre Authorship Verification and Attribution Benchmark”, AAAI, vol. 39, no. 23, pp. 24777-24785, Apr. 2025. 
> GitHub: [https://github.com/mamarcus64/CrossNews](https://github.com/mamarcus64/CrossNews)

## Models
Three transformer architectures were selected to represent different design trade-offs:

| Model | Description | Parameters | Context Length |
|-------|-------------|------------|----------------|
| **DistilBERT** | Lightweight, efficient baseline | 66M | 512 tokens |
| **RoBERTa** | Enhanced BERT with robust pretraining | 125M | 512 tokens |
| **BigBird** | Sparse attention for long sequences | 128M | 4096 tokens |

### Adversarial Attacks
Two LLM-based attack strategies using Flan-T5-Large:

1. **Style Obfuscation**: Untargeted paraphrasing to conceal authorial cues
2. **Style Impersonation**: Targeted rewriting to mimic another author's style

## Results Summary

### In-Domain Performance (Article-Article)
| Model | ROC-AUC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| DistilBERT | 0.8882 | 0.7999 | 0.8161 |
| RoBERTa | 0.8785 | 0.7946 | 0.8084 |
| BigBird | 0.8108 | 0.7321 | 0.7438 |

### Cross-Domain Performance (Article-Tweet)
| Model | ROC-AUC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| DistilBERT | 0.8711 | 0.7874 | 0.8006 |
| RoBERTa | 0.8703 | 0.6127 | 0.4880 |
| BigBird | 0.8149 | 0.6719 | 0.5891 |

### Under Adversarial Attacks (Worst Case: Impersonation + Domain Shift)
| Model | ROC-AUC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| DistilBERT | 0.5590 | 0.5406 | 0.5431 |
| RoBERTa | 0.5587 | 0.5391 | 0.5455 |
| BigBird | 0.5444 | 0.5316 | 0.5305 |

## Interpretability Analysis
**SHAP (SHapley Additive exPlanations)** analysis revealed critical insights:
- Models often rely on **platform-specific artifacts** (hashtags, URLs) rather than genuine stylistic cues
- **Punctuation patterns** and **function words** dominate decisions under adversarial conditions
- Even correct predictions often stem from **topic-related content** rather than authorial style

## Reproducibility
All experiments use fixed random seeds (7, 1001, 1211) for reproducibility.

## Ethics Review
This project has been ethically reviewed and approved by the Ethics Committee of the University of Sheffield.

