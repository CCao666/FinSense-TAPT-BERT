📌 Overview
General-purpose NLP models often struggle with the nuanced vocabulary of the financial sector (e.g., confusing "Shorting" with a negative length or "Bullish" with an animal). This project implements an end-to-end pipeline to adapt a standard BERT model to the financial domain using Task-Adaptive Pretraining (TAPT) and a Calibrated Inference API.

🚀 Key Features
Domain-Adaptive Optimization: Leveraged TAPT (continued Masked Language Modeling) on in-domain financial corpora to align semantic weights before fine-tuning.

Robust Fine-tuning Strategy: Mitigated "Neutral Class Bias" and severe label imbalance using Weighted Cross-Entropy Loss and Label Smoothing.

Production-Ready API: Architected a FastAPI service featuring a Confidence-Margin flagging system to identify and route "Borderline" predictions for manual review.


## 📊 Performance Comparison

The TAPT-enhanced model demonstrates superior stability and minority class recall compared to the baseline BERT-base-uncased. 

| Metric | Baseline BERT | **FinSense-TAPT (Ours)** | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7094 | **0.7350** | +3.6% |
| **F1-Macro** | 0.6277 | **0.6694** | **+6.6%** |
| **Validation Loss** | 0.7594 | 0.8466 | *(Calibrated)* |

> **Note on Loss**: The higher loss in the TAPT model is an intentional result of **Label Smoothing** and **Class Weighting**. This trade-off significantly improves the model's ability to identify minority classes (Positive/Negative) which are often overwhelmed by Neutral samples in baseline models.



🔍 Why it Works: The Calibration Logic
In high-stakes financial environments, a confident mistake is costlier than an admitted uncertainty.

TAPT: Bridges the semantic gap for entities like $AAPL or events like M&A.

Label Smoothing: Prevents the model from becoming over-confident in its predictions, leading to better generalization on unseen news.

Uncertainty Flagging: Our API calculates the Margin (difference between the top two predicted probabilities). If the margin is low, the prediction is flagged as Borderline.

🛠️ API & Usage
The system is deployed via FastAPI and Gradio for real-time inference.
