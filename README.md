🏥 Dependable-Med-LLM (117M)
A custom-built, decoder-only Transformer designed from scratch for medical anomaly detection and highly reliable text generation.

📌 Overview
This repository contains a fully custom, 117-million parameter Large Language Model built entirely from scratch in PyTorch. Unlike standard generative models, this architecture is strictly designed around the concept of Dependable AI. It is specifically optimized for critical medical and military use cases (e.g., Digital Patient Twins) where hallucinations must be prevented at all costs.

The model features a multi-task learning approach, combining standard causal language modeling with a specialized Sequence Classification Head to detect anomalies in medical logs and data streams.

⚙️ Architecture & Technical Highlights
The model implements state-of-the-art transformer mechanics to ensure high memory efficiency and training stability:

Core Architecture: Decoder-only Transformer.

Attention Mechanism: Grouped Query Attention (GQA) for massive memory savings and faster inference.

Activation Function: SwiGLU (Swish-Gated Linear Unit) for superior expressivity in the Feed-Forward Networks.

Normalization: Pre-Norm using RMSNorm to prevent vanishing gradients in deep layers.

Parameter Efficiency: Weight Tying (sharing weights between the embedding layer and the LM head) to save ~24.5M parameters.

Stability: Scaled Weight Initialization (GPT-2 style: scaling residual weights by 1/sqrt(2L)) and 10% Dropout to prevent overfitting.

Custom Tokenizer: Vocabulary size of ~32,000 tokens.

🧬 Multi-Tasking: The Anomaly Head
To support medical data analysis, a custom Multi-Layer Perceptron (MLP) with SiLU activation is attached as an Anomaly Head.

Taking advantage of the causal masking in the decoder, this head performs sequence classification by analyzing the last token of the sequence, ensuring the entire context of the medical log is taken into account before classifying anomalies.

🛡️ Defensive Pipeline against Hallucinations
Generative AI in healthcare requires strict safety guardrails. This repository includes a custom Defensive Pipeline that acts as a supervisory system. It validates the model's outputs before they are presented to the user, strictly reducing the risk of life-threatening hallucinations in clinical scenarios.