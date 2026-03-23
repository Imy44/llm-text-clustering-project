# LLM Text Clustering Project

## 1.Academic Context

This work is part of a university project carried out by Master’s students in Machine Learning for Data Science.

The project is based on the reproduction and analysis of the paper:

**"Text Clustering as Classification with LLMs"**

by:

- Chen Huang — Singapore University of Technology and Design, Singapore  
- Guoxiu He — East China Normal University, Shanghai, China  

⚠️ This repository does **not present original research**, but a **student reproduction and analysis** of the methods proposed in the paper, conducted under practical constraints (API limits, computational resources, and time).

---

## 2.Project Overview

This project focuses on the reproduction and analysis of the paper:

**"Text Clustering as Classification with LLMs"**

The main objective is to study how Large Language Models (LLMs) can be used to transform a clustering problem into a classification task, and to evaluate the reproducibility and robustness of this approach in real-world conditions.

---

## 3.Objectives

- Reproduce the pipeline proposed in the paper  
- Evaluate its performance using standard metrics (ACC, NMI, ARI)  
- Compare LLM-based approaches with classical clustering methods  
- Identify limitations and reproducibility challenges  
- Propose improvements to stabilize the pipeline  

---


Each folder corresponds to a different LLM tested in the project.

---

## 4.Models Used

### 🔹 GPT-3.5
- Original model used in the paper  
- Used for baseline reproduction  
- Issues observed:  
  - Too many generated labels  
  - Poor label merging  
  - Cluster fragmentation  

---

### 🔹 Mistral Large (Final Model ✅)
- Selected as the main model  
- Better stability and structured outputs  
- Improved label merging with constraints  
- Best trade-off between performance and cost  

---

### 🔹 Qwen 3.5 (9B)
- Tested locally (GPU required)  
- Observed issues:  
  - Collapse to a single cluster  
  - Very low performance (ACC ≈ 0, ARI = 0, NMI = 0)  
- Not suitable for this task in our setup  

---

### 🔹 LLaMA 3.1
- High diversity of generated labels  
- Main issue:  
  - Over-clustering (too many clusters)  
  - High computational cost (~6h per run)  

---

## 5.Pipeline Description

The implemented pipeline follows the structure described in the paper:

1. **Few-shot label selection**
   - ~20% of true labels used as examples  

2. **Label generation**
   - LLM generates labels from text batches  

3. **Label merging (critical step)**
   - Merge similar labels to reduce redundancy  

4. **Classification**
   - Assign each text to the most relevant label  

5. **Evaluation**
   - Metrics: Accuracy (ACC), NMI, ARI  

---

## 6.Main Challenges

- API cost (OpenAI / Mistral)  
- Rate limits and token limits  
- Non-deterministic LLM behavior  
- Difficulty in reproducing results from the paper  
- Sensitivity to prompt design  

---

## 7.Proposed Improvements

- Controlled label merging:  
  - Target number of clusters  
  - Avoid over-merging and under-merging  

- Improved JSON parsing robustness  
- More strict classification prompts  
- Better handling of invalid outputs  

---

## 8.Results Summary

- Results are generally consistent with the paper trends  
- Significant improvement with Mistral compared to GPT-3.5  
- Merge step plays a critical role in performance  
- Performance drops on complex datasets (e.g. MTOP)  

---

## 9.Experimental Constraints

- Limited to **1000 samples per dataset**  
- GPU limitations  
- API costs and rate limits  
- Limited number of experimental runs  

---

## 10.Key Takeaways

- LLM-based clustering is promising but still unstable  
- The quality of label generation and merging is crucial  
- Reproducibility remains a major challenge  
- Model choice strongly impacts performance  

---

## 11.Reference

Huang, C., & He, G. (2024).  
**Text Clustering as Classification with LLMs**

---

## 12.Authors

- Imane TAGHI  
- Aida NEISANI  

Master 2 — Machine Learning for Data Science  
Academic Year 2025–2026
