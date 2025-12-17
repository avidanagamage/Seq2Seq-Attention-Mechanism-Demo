# Seq2Seq Attention Mechanism Demo

A comprehensive NLP project demonstrating the impact of the **Attention Mechanism** on Sequence-to-Sequence (Seq2Seq) models. This project compares a baseline GRU Encoder-Decoder against an Attention-based model (Bahdanau Attention) on a character-level string reversal task.

It includes a **Streamlit** web application to visualize predictions and attention heatmaps interactively.

## 📂 Project Structure

```text
NLP Project/
├── data/                  # Dataset generation and utilities
│   ├── generate_dataset.py
│   ├── dataset_utils.py
│   └── char_vocab.json
├── models/                # Model architectures and training scripts
│   ├── baseline/          # Standard Seq2Seq (Encoder-Decoder)
│   └── attention/         # Seq2Seq with Bahdanau Attention
├── demo_app/              # Streamlit Web Application
│   ├── app.py
│   └── sample_inputs.json
└── visualization/         # Scripts to generate training heatmaps
    └── save_attention_heatmaps.py
